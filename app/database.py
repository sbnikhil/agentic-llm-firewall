import os
from astrapy import DataAPIClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')
client = DataAPIClient(os.getenv("ASTRA_DB_APPLICATION_TOKEN"))

KEYSPACE_NAME = os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")
COLLECTION_NAME = os.getenv("ASTRA_DB_COLLECTION", "secure_logs")

db = client.get_database_by_api_endpoint(
    os.getenv("ASTRA_DB_API_ENDPOINT"),
    keyspace=KEYSPACE_NAME 
)

if COLLECTION_NAME not in db.list_collection_names():
    print(f"creating {COLLECTION_NAME} in {KEYSPACE_NAME}...")
    collection = db.create_collection(
        COLLECTION_NAME,
        definition={"vector": {"dimension": 384, "metric": "cosine"}}
    )
else:
    print(f"connected to {COLLECTION_NAME} in {KEYSPACE_NAME}.")
    collection = db.get_collection(COLLECTION_NAME)

def log_interaction(original, safe, response):
    vector = model.encode(original).tolist()    
    collection.insert_one({
        "original_text": original,
        "redacted_text": safe,
        "ai_response": response,
        "$vector": vector 
    })

def leak_check(ai_response_text: str, threshold=0.85):
    """
    Checks if the AI's response is too similar to any known sensitive data.
    Returns (is_leak, similarity_score)
    """
    query_vector = model.encode(ai_response_text).tolist()
    result = collection.find_one(
        {},
        sort={"$vector": query_vector},
        include_similarity=True
    )

    if result and "$similarity" in result:
        score = result["$similarity"]
        if score > threshold:
            return True, score
    
    return False, 0.0