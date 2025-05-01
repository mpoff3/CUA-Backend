import requests
import pandas as pd
from typing import List

def process_inputs(user_inputs: List[str], server_url: str = "http://localhost:8000") -> None:
    # Initialize lists to store responses
    final_responses = []
    num_commands_list = []
    num_command_types_list = []
    
    # Process each input
    for user_input in user_inputs:
        print(f"\nProcessing input: {user_input}")
        try:
            # Make POST request to server
            response = requests.post(
                f"{server_url}/process",
                json={"user_input": user_input}
            )
            response.raise_for_status()  # Raise exception for bad status codes
            
            # Extract data from response
            data = response.json()
            final_responses.append(data["final_response"])
            num_commands_list.append(data["num_commands"])
            num_command_types_list.append(data["num_command_types"])
            
            print("Successfully processed input")
            
        except Exception as e:
            print(f"Error processing input: {str(e)}")
            # Append None or error message to maintain array alignment
            final_responses.append(str(e))
            num_commands_list.append(None)
            num_command_types_list.append(None)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame({
        'user_input': user_inputs,
        'final_response': final_responses,
        'num_commands': num_commands_list,
        'num_command_types': num_command_types_list
    })
    
    # Save to CSV
    output_file = 'processing_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    test_inputs = [
        "Search for a white couch and tell me the first one you see",
        "Search for a blue couch and tell me the first one you see",
    ]
    
    process_inputs(test_inputs) 