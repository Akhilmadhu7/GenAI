from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()

client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

def get_embedding(text, model="text-embedding-3-small"):
    """
    Generate embedding for text using OpenAI API
    
    Args:
        text: Input text string
        model: Embedding model to use
    Returns:
        List of floats (the embedding vector)
    """
    input_text = text.replace("\n", " ").strip()
    response = client.embeddings.create(
        input=[input_text],
        model=model
    )
    return response.data[0].embedding


def cosine_similarity(vecA, vecB):
    vec1 = np.array(vecA)
    vec2 = np.array(vecB)
    dot_product = np.dot(vec1, vec2)
    normA = np.linalg.norm(vec1)
    normB = np.linalg.norm(vec2)
    if normA == 0 or normB == 0:
        return 0.0
    else:
        return dot_product / (normA * normB)


def rag_query(query, documents, top_k=3):

    print("\n[Step 1] Converting query to embedding...")
    query_embedding = get_embedding(query)
    print(f"✓ Query embedding generated ({len(query_embedding)} dimensions)")

    print(f"\n[Step 2] Searching {len(documents)} documents...")
    doc_similarities = []
    for doc in documents:
        doc_embedding = get_embedding(doc)
        similarity = cosine_similarity(query_embedding, doc_embedding)
        doc_similarities.append((doc, similarity))

    # Sort by similarity
    doc_similarities.sort(key=lambda x: x[1], reverse=True)
    top_results = doc_similarities[:top_k]

    print(f" Found {top_k} most relevant documents:")
    for i, (doc, score) in enumerate(top_results, 1):
        print(f"\n  {i}. Similarity: {score:.4f}")
        print(f"     {doc[:80]}...")
    
    print(f"\n[Step 3] Building context from retrieved documents...")
    context = "\n\n".join([doc for doc, _ in top_results])
    print("context: ", context)
    print(f"✓ Context prepared ({len(context)} characters)")

    prompt = f"""Answer the question based on the provided context.

            <context>
            {context}
            </context>

            Question: {query}

            Answer based only on the context provided:
        """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user", "content":prompt}],
        max_tokens=1024
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    # Sample knowledge base
    knowledge_base = [
        "Acme Corp offers 15 days of paid vacation annually. Employees must submit vacation requests at least 2 weeks in advance.",
        
        "Remote work policy: Employees can work from home up to 3 days per week with manager approval.",
        
        "Health insurance covers medical, dental, and vision. Employees contribute 20% of premiums.",
        
        "The equipment stipend is $500 per year for home office setup. Submit receipts to HR for reimbursement.",
        
        "Professional development budget: $1,000 annually for courses, conferences, or certifications.",
    ]
    
    # Test queries
    queries = [
        "How many vacation days do employees get?",
        # "What's the remote work policy?",
        # "Tell me about the equipment budget"
    ]

    for query in queries:
        print(f"\n=== Query: {query} ===")
        answer = rag_query(query, knowledge_base)
        print(f"\nAnswer: {answer}\n")
                