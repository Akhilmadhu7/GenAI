import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))


# Our "knowledge base" - just a simple string for now
COMPANY_POLICY = """
Acme Corp Remote Work Policy (Updated 2024):
- Employees can work remotely up to 3 days per week.
- Remote work must be approved by direct managers in advance.
- All employees must attend in-person meetings on Wednesdays.
- Remote workers must be available during core hours: 10 AM - 3 PM EST.
- Equipment stipend: $500 annually for home office setup.
"""

def ask_with_context(question, context):
    
    # Construct prompt with context
    prompt = f"""Here is some context information:

        <context>
        {context}
        </context>

        Based ONLY on the context above, please answer this question: {question}

        If the answer cannot be found in the context, say "I don't have that information in the provided context."
    """

    messages = [
        {"role":"user", "content":prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1024
    )
    print("Model response: ", response, "\n")

    response_message = response.choices[0].message
    print("User Question: ", question, "\n")
    print("Model Answer: ", response_message.content, "\n")
    return response_message.content


if __name__ == "__main__":
    # Example question that can be answered from the context
    ask_with_context("How many days a week can Acme Corp employees work remotely?", COMPANY_POLICY)

    # Example question that cannot be answered from the context
    ask_with_context("What is the process for requesting time off at Acme Corp?", COMPANY_POLICY)