from dotenv import load_dotenv
from openai import OpenAI
import os

def get_stream_resonse():
    client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        temperature=0.7,
        max_tokens=23,
        stream=True
    )

    full_response = ""
    for chunk in stream:
        print("Chunk: ", chunk)
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
        
        # print("Chunk content: ", chunk.choices[0].delta.get("content", ""))
    return full_response

if __name__ == "__main__":
    load_dotenv()
    get_stream_resonse()


