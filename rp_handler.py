import os
import runpod
import time  

import ollama  # Ollama client

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
# MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3:8b")


# Initialize the Ollama client
ollama_base_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
print(f"🔗 Connecting to Ollama at: {ollama_base_url}")
client = ollama.Client(host=ollama_base_url)

def process_request(prompt: str):
    """
    Process a request to the Ollama service.
    
    Args:
        prompt (str): The prompt to send to the Ollama service
    """
    response = client.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    return response.message.content

def handler(event):
#   This function processes incoming requests to your Serverless endpoint.
#
#    Args:
#        event (dict): Contains the input data and request metadata
#       
#    Returns:
#       Any: The result to be returned to the client
    
    # Extract input data
    print(f"Worker Start")
    input = event['input']
    
    prompt = input.get('prompt')  
    seconds = input.get('seconds', 0)  

    print(f"Received prompt: {prompt}")
    print(f"Sleeping for {seconds} seconds...")
    
    # You can replace this sleep call with your own Python code
    time.sleep(seconds) 

    response = process_request(prompt)
    print(f"Response: {response}")
    
    return response 

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })
