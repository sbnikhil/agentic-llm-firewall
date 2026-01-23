import os
import json
import numpy as np
from pathlib import Path
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
    collection = db.create_collection(
        COLLECTION_NAME,
        definition={"vector": {"dimension": 384, "metric": "cosine"}}
    )
else:
    collection = db.get_collection(COLLECTION_NAME)

def log_interaction(original, safe, response):
    vector = model.encode(original).tolist()    
    collection.insert_one({
        "original_text": original,
        "redacted_text": safe,
        "ai_response": response,
        "$vector": vector 
    })

def load_secrets_catalog():
    catalog_path = Path(__file__).parent.parent / "secrets_catalog.json"
    try:
        with open(catalog_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def leak_check(ai_response_text: str, threshold=0.85):
    secrets_catalog = load_secrets_catalog()
    
    if not secrets_catalog:
        return False, 0.0
    
    query_vector = model.encode(ai_response_text).tolist()
    
    for secret_entry in secrets_catalog:
        secret_text = secret_entry.get("text", "")
        secret_vector = model.encode(secret_text).tolist()
        import numpy as np
        response_vec = np.array(query_vector)
        secret_vec = np.array(secret_vector)
        
        similarity = np.dot(response_vec, secret_vec) / (
            np.linalg.norm(response_vec) * np.linalg.norm(secret_vec)
        )
        
        if similarity > threshold:
            return True, float(similarity)
    
    return False, 0.0