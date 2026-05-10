from chromadb import Client
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
import os


load_dotenv()

client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
chroma_client = chromadb.Client()


#create collection
collection = chroma_client.create_collection(
    name="documentation",
    metadata={"description": "A collection of security profiles for different users"}
)
print("✓ Created collection in vector database!\n")

def get_embeddings(text:str, model:str="text-embedding-3-small"):
    response = client.embeddings.create(
        input=[text.replace("\n", " ")],
        model=model
    )
    return response.data[0].embedding

# Sample documentation
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

# Metadata for each document
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

print("Generating embeddings and storing documents...")

embeddings = [get_embeddings(doc) for doc in documents]

collection.add(
    embeddings=embeddings,
    metadatas=metadata,
    documents=documents,
    ids=[f"doc_{len(i)}" for i in documents]
)

print(f"✓ Stored {len(documents)} documents in vector database!")

def semantic_search(query:str, n_results:int = 3):
    embedding = get_embeddings(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results
    )
    return results

# Test queries
test_queries = [
    "How can I change my login credentials?",  # Should find password doc
    "I want to remove my profile permanently",  # Should find account deletion
    "How do I get my information from the platform?"  # Should find data export
]

for query in test_queries:
    print(f"\nQuery: {query}")
    results = semantic_search(query)
    print("Top results:", results,"\n")
    for i, (doc,meta,distance) in enumerate(zip(results['documents'][0], results['metadatas'][0],results['distances'][0])):
        print(f" Result {i+1}: {doc} (Category: {meta['category']}, Topic: {meta['topic']}, Distance: {distance})")