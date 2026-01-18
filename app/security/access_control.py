from config import SecurityConfig
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

TOPIC_EXAMPLES = {
    "public": [
        "remote work policy",
        "office hours",
        "company benefits",
        "vacation policy"
    ],
    "internal": [
        "project timelines",
        "team structure",
        "equipment requests",
        "expense approval"
    ],
    "confidential": [
        "budget information",
        "financial data",
        "strategic plans",
        "executive decisions"
    ],
    "secret": [
        "passwords",
        "access codes",
        "api keys",
        "credentials"
    ]
}

topic_embeddings = {
    level: [model.encode(example) for example in examples]
    for level, examples in TOPIC_EXAMPLES.items()
}

def classify_query_sensitivity(query: str) -> str:
    query_embedding = model.encode(query)
    
    scores = {}
    for level, embeddings in topic_embeddings.items():
        similarities = [
            np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            for emb in embeddings
        ]
        scores[level] = max(similarities)
    
    if scores["secret"] > 0.7:
        return "secret"
    elif scores["confidential"] > 0.65:
        return "confidential"
    elif scores["internal"] > 0.6:
        return "internal"
    else:
        return "public"

def has_permission(user_role: str, topic_level: str) -> bool:
    allowed_levels = SecurityConfig.ROLE_PERMISSIONS.get(user_role, ["public"])
    return topic_level in allowed_levels

def get_allowed_collections(user_role: str) -> list[str]:
    allowed_levels = SecurityConfig.ROLE_PERMISSIONS.get(user_role, ["public"])
    return [
        SecurityConfig.COLLECTION_NAMES[level]
        for level in allowed_levels
        if level in SecurityConfig.COLLECTION_NAMES
    ]
