from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from agent.agent import Agent
from computers import LocalPlaywrightComputer
from collections import Counter
import logging

app = FastAPI()

class UserInput(BaseModel):
    user_input: str

class Response(BaseModel):
    final_response: str
    num_commands: int
    num_command_types: int

def process_user_input_single_attempt(user_input: str) -> Dict:
    with LocalPlaywrightComputer(headless=False) as computer:
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
                print_steps=True,  # Enable printing of reasoning and actions
                show_images=False,
                debug=False,
                reasoning_summary="concise"
            )
            
            # Process all items from this turn
            for item in output_items:
                items.append(item)
                # Only set has_final_response after processing all items
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
            # Handle the case where content is a list of message objects
            content = final_message.get("content", [])
            if isinstance(content, list) and len(content) > 0:
                # Extract text from the first message object
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
def process_request(user_input: UserInput) -> Response:
    try:
        result = process_user_input(user_input.user_input)
        return Response(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 