from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
from agent.agent import Agent
from computers import LocalPlaywrightComputer
from collections import Counter
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial
from contextlib import asynccontextmanager
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    import asyncio
    app.state.loop = asyncio.get_event_loop()
    yield
    # Shutdown code
    process_pool.shutdown(wait=True)

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Use a ProcessPoolExecutor, forcing the "spawn" start method so that
# worker processes start with a **fresh** Python interpreter instead of being
# forked.  On Linux the default is "fork", which copies the parent asyncio
# event-loop into the child and makes Playwright (sync API) think there is an
# already-running loop, resulting in the runtime error you observed inside
# Docker.  Using "spawn" matches the behaviour you get on macOS and Windows
# where the code already works.
process_pool = ProcessPoolExecutor(max_workers=10, mp_context=multiprocessing.get_context("spawn"))

class UserInput(BaseModel):
    user_input: str
    product_search: str

class Response(BaseModel):
    final_response: str
    num_commands: int
    num_command_types: int
    score: int
    reasoning: str

def evaluate_response_sync(input_text: str, response_text: str) -> dict:
    """Synchronously evaluate a single input-response pair using GPT-4o"""
    EVALUATION_PROMPT = """You are an expert evaluator. Your task is to evaluate how well a response matches its input query.\nGiven an input query and the response, rate the match on a scale from 1 to 10, where:\n1 = Completely irrelevant or incorrect response\n10 = Perfect match that directly answers the query\n\nInput Query: {input}\nResponse: {response}\n\nProvide your evaluation in the following format:\nScore: [1-10]\nReasoning: [Your explanation]\n\nRemember:\n- Focus on how well the response addresses the specific query\n- Consider accuracy and relevance\n- Ignore minor grammatical issues\n"""
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": EVALUATION_PROMPT.format(
                    input=input_text,
                    response=response_text
                )}
            ]
        )
        evaluation = completion.choices[0].message.content
        score_line = [line for line in evaluation.split('\n') if line.startswith('Score:')][0]
        reasoning_line = [line for line in evaluation.split('\n') if line.startswith('Reasoning:')][0]
        score = int(score_line.split(':')[1].strip())
        reasoning = reasoning_line.split(':', 1)[1].strip()
        return {"score": score, "reasoning": reasoning}
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return {"score": None, "reasoning": f"Evaluation error: {str(e)}"}

def run_playwright_process(user_input: str, product_search: str) -> Dict:
    """Run the Playwright code in a completely separate process with automatic retries."""
    return process_user_input(user_input, product_search)

def process_user_input_single_attempt(user_input: str, product_search: str) -> Dict:
    with LocalPlaywrightComputer(headless=True) as computer:
        try:
            # Initialize browser with starting URL
            computer.goto("https://www.wayfair.com")
            
            agent = Agent(
                model="computer-use-preview",
                computer=computer,
                acknowledge_safety_check_callback=lambda x: True  # Auto-approve safety checks
            )
            items = []
            has_final_response = False
            
            # Initialize the conversation with user input
            items.append({"role": "user", "content": user_input})
            
            # Run the agent and wait for final response
            while not has_final_response:
                output_items = agent.run_full_turn(
                    items,
                    print_steps=True,
                    show_images=False,
                    debug=False,
                    reasoning_summary="concise"
                )
                
                # Process all items from this turn
                for item in output_items:
                    items.append(item)
                    if item.get("role") == "assistant":
                        has_final_response = True
            
            # Get the final assistant message
            final_message = next(
                (item for item in reversed(items) 
                 if item.get("role") == "assistant"),
                None
            )
            
            if final_message is None:
                final_response = "No response from assistant"
            else:
                content = final_message.get("content", [])
                if isinstance(content, list) and len(content) > 0:
                    final_response = content[0].get("text", "No text in response")
                else:
                    final_response = str(content)
            
            # Count computer actions
            computer_calls = [
                item for item in items 
                if item.get("type") == "computer_call"
            ]
            
            num_commands = len(computer_calls)
            action_types = Counter(
                call["action"]["type"] 
                for call in computer_calls
            )
            num_command_types = len(action_types)
            
            # Evaluate response
            evaluation = evaluate_response_sync(product_search, final_response)
            
            return {
                "final_response": final_response,
                "num_commands": num_commands,
                "num_command_types": num_command_types,
                "score": evaluation["score"],
                "reasoning": evaluation["reasoning"]
            }
        except Exception as e:
            print(f"Error during attempt: {str(e)}")
            return {"final_response": f"Error processing user input: {str(e)}", "num_commands": 0, "num_command_types": 0, "score": None, "reasoning": f"Evaluation error: {str(e)}"}

def process_user_input(user_input: str, product_search: str, max_retries: int = 2) -> Dict:
    """Process user input with automatic retries on failure."""
    attempt = 0
    last_result = None

    while attempt < max_retries:
        print(f"\nAttempt {attempt + 1} of {max_retries}")
        result = process_user_input_single_attempt(user_input, product_search)
        final_response = result.get("final_response", "")
        # If the attempt produced an error indicator, retry
        if "error" in final_response.lower():
            attempt += 1
            last_result = result
            print(f"Error during attempt {attempt}: {final_response}")
            print("Retrying with a fresh browser session...")
            continue
        # Successful response
        return result

    # All attempts failed; return the last error result
    return last_result

@app.post("/process", response_model=Response)
async def process_request(user_input: UserInput) -> Response:
    try:
        # Run the synchronous processing code in a separate process
        result = await app.state.loop.run_in_executor(
            process_pool,
            run_playwright_process,
            user_input.user_input,
            user_input.product_search
        )
        return Response(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Required for Windows compatibility
    multiprocessing.freeze_support()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 