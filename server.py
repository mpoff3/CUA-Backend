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

app = FastAPI()

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

class Response(BaseModel):
    final_response: str
    num_commands: int
    num_command_types: int

def run_playwright_process(user_input: str) -> Dict:
    """Run the Playwright code in a completely separate process"""
    return process_user_input(user_input)

def process_user_input_single_attempt(user_input: str) -> Dict:
    with LocalPlaywrightComputer(headless=True) as computer:
        try:
            # Initialize browser with starting URL
            computer.goto("https://www.westelm.com")
            
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
            
            return {
                "final_response": final_response,
                "num_commands": num_commands,
                "num_command_types": num_command_types
            }
        except Exception as e:
            print(f"Error during attempt: {str(e)}")
            return {"final_response": f"Error processing user input: {str(e)}", "num_commands": 0, "num_command_types": 0}

def process_user_input(user_input: str, max_retries: int = 3) -> Dict:
    """Process user input with automatic retries on failure."""
    attempt = 0
    last_error = None
    
    while attempt < max_retries:
        try:
            print(f"\nAttempt {attempt + 1} of {max_retries}")
            return process_user_input_single_attempt(user_input)
        except Exception as e:
            last_error = e
            attempt += 1
            print(f"Error during attempt {attempt}: {str(e)}")
            print("Retrying with a fresh browser session...")
            
    # If we've exhausted all retries, raise the last error
    raise last_error or Exception("Failed to process user input after all retries")

@app.post("/process", response_model=Response)
async def process_request(user_input: UserInput) -> Response:
    try:
        # Run the synchronous processing code in a separate process
        result = await app.state.loop.run_in_executor(
            process_pool,
            run_playwright_process,
            user_input.user_input
        )
        return Response(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    # Store the event loop for use in endpoints
    import asyncio
    app.state.loop = asyncio.get_event_loop()

@app.on_event("shutdown")
async def shutdown_event():
    # Clean up the process pool
    process_pool.shutdown(wait=True)

if __name__ == "__main__":
    # Required for Windows compatibility
    multiprocessing.freeze_support()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 