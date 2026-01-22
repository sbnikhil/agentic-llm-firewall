import os
import logging
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def ingest():
    # Define documents with access level metadata
    documents = [
        # PUBLIC
        Document(
            page_content="Global Corp Remote Work Policy: Employees are allowed to work from home 3 days a week. All equipment requests must be submitted through the 'EquipMe' portal. Standard office hours are 9 AM to 5 PM with flexible scheduling. Company provides health insurance, 401k matching, and wellness benefits.",
            metadata={"access_level": "public", "source": "company_policy"}
        ),
        # INTERNAL
        Document(
            page_content="Equipment and Travel Guidelines: Equipment requests are processed within 5 business days. Travel expenses over $500 require VP approval. Project timelines and team assignments are available in the internal wiki.",
            metadata={"access_level": "internal", "source": "company_policy"}
        ),
        # CONFIDENTIAL
        Document(
            page_content="Financial Guidelines for Management: Department budgets are reviewed quarterly. Strategic planning documents are restricted to management level and contain sensitive financial projections.",
            metadata={"access_level": "confidential", "source": "company_policy"}
        ),
    ]
    
    logger.info(f"Prepared {len(documents)} documents with access level metadata")

    logger.info("Connecting to Astra DB Vector Store...")
    try:
        vstore = AstraDBVectorStore(
            embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            collection_name="company_knowledge",
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
            token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
        )
        logger.info("Connection successful")
        
        # Clear existing data
        logger.info("Clearing existing data...")
        vstore.clear()
        
    except Exception as e:
        logger.error(f"Connection error: {e}")
        return

    logger.info("Uploading documents...")
    vstore.add_documents(documents)
    logger.info("Successfully ingested documents with access level metadata")

if __name__ == "__main__":
    ingest()
else:
    logger.warning("Script was imported, not run directly")