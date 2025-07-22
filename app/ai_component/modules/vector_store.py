import sys
import os
import json
from datetime import datetime
from typing import List, Dict, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import Qdrant
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.ai_component.config import top_collection_search
from app.ai_component.logger import logging
from app.ai_component.exception import CustomException
from dotenv import load_dotenv

load_dotenv()

class LongTermMemory: 
    def __init__(self, qdrant_url: str = os.getenv("QDRANT_URL"), google_api_key: str = os.getenv("GOOGLE_API_KEY")):
        self.qdrant_url = qdrant_url
        self.google_api_key = google_api_key

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.google_api_key
        )

        self.client = QdrantClient(
            url=self.qdrant_url,
            prefer_grpc=False
        )

    def _collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists"""
        try:
            collections = self.client.get_collections()
            return any(collection.name == collection_name for collection in collections.collections)
        except Exception as e:
            logging.error(f"Error checking collection existence: {str(e)}")
            return False

    def create_collection(self, collection_name: str, vector_size: int = 768) -> bool:
        """Create new collection"""
        try:
            # Check if collection already exists
            if self._collection_exists(collection_name):
                logging.info(f"Collection {collection_name} already exists")
                return True
                
            logging.info("Creating new collection")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            logging.info("New collection created")
            return True
        except CustomException as e:
            logging.error(f"Error in creating collection; {str(e)}")
            raise CustomException(e, sys) from e

    def load_json_file(self, file_path: str) -> List[Document]:
        """Load and parse specific JSON file (gut_health_qa_pairs.json)"""
        try:
            documents = []
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File {file_path} not found")
            
            logging.info(f"Processing file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                if isinstance(json_data, list):
                    data_items = json_data
                else:
                    data_items = [json_data]
                
                for item in data_items:
                    if 'question' not in item or 'answer' not in item:
                        continue
                    
                    content = f"Question: {item['question']}\n\nAnswer: {item['answer']}"
                    
                    metadata = {
                        'source': os.path.basename(file_path)
                    }
                    
                    # Create Document object
                    doc = Document(
                        page_content=content,
                        metadata=metadata
                    )
                    documents.append(doc)
            
            logging.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logging.error(f"Error loading JSON file: {str(e)}")
            raise CustomException(e, sys) from e

    async def StoreInMemory(self, collection_name: str, file_path: str, chunk_size: int = 2000, chunk_overlap: int = 100) -> bool:
        """
        Store the specific JSON file data in the vector database
        """
        try:
            logging.info(f"Storing JSON data from {file_path}")
            
            # Load JSON document
            documents = self.load_json_file(file_path)
            
            if not documents:
                logging.warning("No documents found to store")
                return False
            
            logging.info(f"Processing {len(documents)} Q&A pairs for collection {collection_name}")
            
            self.create_collection(collection_name)
            
            ### Store documents directly without splitting (each Q&A pair as one document)
            ### Only split if individual Q&A pairs are too large
            texts_to_store = []
            for doc in documents:
                if len(doc.page_content) > chunk_size:
                    ### Only split if the Q&A pair is very large
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    split_docs = text_splitter.split_documents([doc])
                    texts_to_store.extend(split_docs)
                else:
                    texts_to_store.append(doc)
            
            qdrant = Qdrant.from_documents(
                texts_to_store,
                self.embeddings,
                url=self.qdrant_url,
                collection_name=collection_name,
                prefer_grpc=False
            )
            
            logging.info(f"Successfully stored {len(texts_to_store)} documents in collection {collection_name}")
            return True
            
        except CustomException as e:
            logging.error(f"Error in JSON storing: {str(e)}")
            raise CustomException(e, sys) from e

    def search_in_collection(self, query: str, collection_name: str, k: int = top_collection_search) -> List:
        """Search in the collection"""
        try:
            if not self._collection_exists(collection_name=collection_name):
                logging.warning(f"Collection {collection_name} does not exist")
                return []
                
            logging.info("Search in collection")
            db = Qdrant(
                client=self.client,
                collection_name=collection_name,
                embeddings=self.embeddings
            )
            docs = db.similarity_search_with_score(query=query, k=k)
            logging.info("Relevant docs found with score")
            return docs
            
        except CustomException as e:
            logging.error(f"Error in similarity search {str(e)}") 
            raise CustomException(e, sys) from e

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            if self._collection_exists(collection_name):
                self.client.delete_collection(collection_name)
                logging.info(f"Collection {collection_name} deleted")
                return True
            else:
                logging.warning(f"Collection {collection_name} does not exist")
                return False
        except Exception as e:
            logging.error(f"Error deleting collection: {str(e)}")
            raise CustomException(e, sys) from e

memory = LongTermMemory()  

if __name__ == "__main__":
    import asyncio
    
    async def main():
        memory = LongTermMemory()
        collection_name = "gut_health_qa_collection"
        file_path = r"alldata\gut_health_qa_pairs.json"  
        success = await memory.StoreInMemory(collection_name, file_path)
        
        if success:
            query = "What are the symptoms of microbiome?"
            results = memory.search_in_collection(query, collection_name, k=3)
            
            print(f"Search results for: '{query}'")
            for i, (doc, score) in enumerate(results):
                print(f"\nResult {i+1} (Score: {score:.4f}):")
                print(f"Content: {doc.page_content[:300]}...")
                print(f"Metadata: {doc.metadata}")
        else:
            print("Failed to store data")
    
    asyncio.run(main())