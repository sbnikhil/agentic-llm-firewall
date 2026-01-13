import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

def ingest():
    try:
        loader = TextLoader("company_policy.txt")
        documents = loader.load()
        print(f"loaded {len(documents)} document(s).")
    except Exception as e:
        print(f"error loading file: {e}")
        return

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    print(f"split into {len(docs)} chunks.")

    print("connecting to Astra DB Vector Store...")
    try:
        vstore = AstraDBVectorStore(
                    embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
                    collection_name="company_knowledge",
                    api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
                    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
            )
        print("connection successful.")
    except Exception as e:
        print(f"connection error: {e}")
        return

    print("uploading chunks to the cloud")
    vstore.add_documents(docs)
    print("success!")

if __name__ == "__main__":
    ingest()
else:
    print("script was imported, not run directly.")