import chromadb
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

chroma_client = chromadb.Client()

collection = chroma_client.create_collection(
    name="enfin",
    metadata={"description":""}
)

def get_embeddings(text:str, model:str="text-embedding-3-small"):
    response = client.embeddings.create(
        input=[text.replace("\n", " ")],
        model=model
    )
    return response.data[0].embedding


def semantic_search(query:str, n_results:int = 3):
    
    query_embedding = get_embeddings(query)
    print("query embedding ",query_embedding)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results

documents = [
    "To reset your password, go to Settings > Security > Change Password. Enter your current password and then your new password twice.",
    "You can update your email address in the Account section. Click on Profile, then Edit Email, and verify the change via the confirmation link.",
    "To delete your account, navigate to Settings > Privacy > Delete Account. This action is permanent and cannot be undone.",
    "Enable two-factor authentication in Security settings. You'll need a mobile app like Google Authenticator or Authy.",
    "Export your data by going to Settings > Data & Privacy > Download Data. Processing may take up to 48 hours.",
    "Change your username in Profile settings. Note that usernames must be unique and can only be changed once every 30 days.",
    "To recover a deleted item, check your Trash folder within 30 days. After 30 days, items are permanently removed.",
    "Manage notification preferences in Settings > Notifications. You can customize alerts for email, push, and SMS."
]

metadata = [
    {"category": "security", "topic": "password"},
    {"category": "account", "topic": "email"},
    {"category": "account", "topic": "deletion"},
    {"category": "security", "topic": "2fa"},
    {"category": "privacy", "topic": "data-export"},
    {"category": "account", "topic": "username"},
    {"category": "recovery", "topic": "trash"},
    {"category": "settings", "topic": "notifications"}
]

print("Generating embeddings and storing documents... \n")

document_embeddings = [get_embeddings(doc) for doc in documents]
collection.add(
    embeddings=document_embeddings,
    metadatas=metadata,
    ids=[f"doc_{len(i)}" for i in documents],
    documents=documents
)

print("Added document embedding. \n")



def ask_question(user_query:str, n_results:int=3):

    """
        Complete RAG pipeline:
        1. Take user question
        2. Generate embedding
        3. Retrieve relevant documents
        4. Create context
        5. Generate answer with LLM
    """

    print(f"Searching knowledge base for: '{user_query}'")
    result = semantic_search(user_query, n_results)
    for r in result:
        print(f"r: {r} \n")
    context_docs = result['documents'][0]

    context = "\n\n".join([f"Document {i+1}: {document}" for i, document in enumerate(context_docs)])
    print("context: \n", context_docs,"\n")

    print(f" Found {len(context_docs)} relevant documents")

    system_prompt = """
        You are a helpful assistant. Answer the user's question 
        using ONLY the information provided in the context below. If the answer cannot 
        be found in the context, say so clearly. Do not make up information
    """
    user_prompt = f"""
        Context: {context}
        Question: {user_query}
        Answer:             
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system", "content":system_prompt},
            {"role":"user", "content":user_prompt}
        ],
        temperature=0.7,
        max_tokens=1024
    )
    return {
        "answer": response.choices[0].message.content,
        "sources": context_docs,
        "metadata": result['metadatas'][0]
    }

test_question = "What should I do if I accidentally deleted something important?" 

result = ask_question(test_question)

print("\n" + "-"*60)
print(f"Question: {test_question}")
print("-"*60)
print(f"\nAnswer:\n{result['answer']}")
print("\n📚 Sources used:")
for i, (source, meta) in enumerate(zip(result['sources'], result['metadata']), 1):
    print(f"\n{i}. [{meta['category']}] {source[:80]}...")
    