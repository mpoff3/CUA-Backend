import aiohttp
import asyncio
import pandas as pd
from typing import List
import os
from openai import AsyncOpenAI

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

EVALUATION_PROMPT = """You are an expert evaluator. Your task is to evaluate how well a response matches its input query.
Given an input query and the response, rate the match on a scale from 1 to 10, where:
1 = Completely irrelevant or incorrect response
10 = Perfect match that directly answers the query

Input Query: {input}
Response: {response}

Provide your evaluation in the following format:
Score: [1-10]
Reasoning: [Your explanation]

Remember:
- Focus on how well the response addresses the specific query
- Consider accuracy and relevance
- Ignore minor grammatical issues
"""

async def evaluate_response(input_text: str, response_text: str) -> dict:
    """Evaluate a single input-response pair using GPT-4o"""
    try:
        completion = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": EVALUATION_PROMPT.format(
                    input=input_text,
                    response=response_text
                )}
            ]
        )
        
        evaluation = completion.choices[0].message.content
        
        # Parse the evaluation to extract score and reasoning
        score_line = [line for line in evaluation.split('\n') if line.startswith('Score:')][0]
        reasoning_line = [line for line in evaluation.split('\n') if line.startswith('Reasoning:')][0]
        
        score = int(score_line.split(':')[1].strip())
        reasoning = reasoning_line.split(':')[1].strip()
        
        return {
            "score": score,
            "reasoning": reasoning
        }
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return {
            "score": None,
            "reasoning": f"Evaluation error: {str(e)}"
        }

async def make_request(session, user_input: str, server_url: str) -> dict:
    """Make a single async request to the server"""
    try:
        async with session.post(
            f"{server_url}/process",
            json={"user_input": user_input},
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        ) as response:
            response.raise_for_status()
            return await response.json()
    except Exception as e:
        print(f"Error processing input '{user_input}': {str(e)}")
        return {
            "final_response": str(e),
            "num_commands": None,
            "num_command_types": None
        }

async def process_inputs_async(user_inputs: List[str], product_search: List[str], server_url: str = "http://0.0.0.0:8001") -> None:
    async with aiohttp.ClientSession() as session:
        # Create tasks for all inputs
        tasks = [make_request(session, input_text, server_url) for input_text in user_inputs]
        
        # Execute all requests in parallel
        results = await asyncio.gather(*tasks)
        
        # Extract data from results
        final_responses = []
        num_commands_list = []
        num_command_types_list = []
        scores = []
        reasonings = []
        
        # Evaluate each input-response pair
        for ps, result in zip(product_search, results):
            final_responses.append(result["final_response"])
            num_commands_list.append(result["num_commands"])
            num_command_types_list.append(result["num_command_types"])
            
            # Evaluate the response
            evaluation = await evaluate_response(ps, result["final_response"])
            scores.append(evaluation["score"])
            reasonings.append(evaluation["reasoning"])
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame({
            'user_input': user_inputs,
            'product_search': product_search,
            'final_response': final_responses,
            'num_commands': num_commands_list,
            'num_command_types': num_command_types_list,
            'match_score': scores,
            'evaluation_reasoning': reasonings
        })
        
        # Save to CSV
        output_file = 'processing_results.csv'
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            df = pd.concat([existing_df, df], ignore_index=True)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        # Print evaluation summary
        print("\nEvaluation Summary:")
        for i, (input_text, ps, final_response, score, reasoning) in enumerate(zip(user_inputs, product_search, final_responses, scores, reasonings), 1):
            print(f"\nQuery {i}:")
            print(f"Input: {input_text}")
            print(f"Product Search: {ps}")
            print(f"Final Response: {final_response}")
            print(f"Score: {score}/10")
            print(f"Reasoning: {reasoning}")

def process_inputs(user_inputs: List[str], product_search: List[str], server_url: str = "http://127.0.0.1:8001") -> None:
    """Wrapper function to run the async code"""
    asyncio.run(process_inputs_async(user_inputs, product_search, server_url))

if __name__ == "__main__":

    df = pd.read_csv('cua_inputs.csv')
    cua_inputs = df['cua_input'].tolist()
    product_search = df['product_search'].tolist()

    print(f'Processing: {cua_inputs[0:1]} and {product_search[0:1]}')

    process_inputs(cua_inputs[0:1], product_search[0:1]) 