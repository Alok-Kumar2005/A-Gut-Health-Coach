import sys
import os
import json
import pickle
from datetime import datetime
from typing import List, Dict, Optional, Union
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import Qdrant
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from app.ai_component.config import top_collection_search
from app.ai_component.logger import logging
from app.ai_component.exception import CustomException
from dotenv import load_dotenv
import time

load_dotenv()

class DataStore: 
    def __init__(self, qdrant_url: str = os.getenv("QDRANT_URL"), google_api_key: str = os.getenv("GOOGLE_API_KEY")):
        self.qdrant_url = qdrant_url
        self.google_api_key = google_api_key
        self.bm25_retriever = None
        self.vector_retriever = None
        self.ensemble_retriever = None
        self.embeddings = None
        self.client = None
        self._initialize_components()

    def _initialize_components(self):
        """Initialize components with proper error handling"""
        try:
            # Initialize embeddings with retry logic
            self._initialize_embeddings()
            
            # Initialize Qdrant client
            self.client = QdrantClient(url=self.qdrant_url, prefer_grpc=False)
            logging.info("DataStore components initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing DataStore: {str(e)}")
            raise CustomException(e, sys) from e

    def _initialize_embeddings(self, max_retries=3):
        """Initialize embeddings with retry logic"""
        for attempt in range(max_retries):
            try:
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=self.google_api_key
                )
                # Test the embeddings with a simple query
                test_embedding = self.embeddings.embed_query("test")
                if test_embedding:
                    logging.info("Embeddings initialized successfully")
                    return
            except Exception as e:
                logging.warning(f"Embeddings initialization attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise Exception(f"Failed to initialize embeddings after {max_retries} attempts: {str(e)}")

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
        except Exception as e:
            logging.error(f"Error in creating collection: {str(e)}")
            raise CustomException(e, sys) from e

    def load_json_file(self, file_path: str) -> List[Document]:
        """Load and parse JSON file with new structure (articles with sections)"""
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
                    if 'sections' not in item or not isinstance(item['sections'], list):
                        continue
                    
                    article_source = item.get('source', 'unknown')
                    article_url = item.get('url', '')
                    article_title = item.get('title', '')
                    
                    for section in item['sections']:
                        if not isinstance(section, dict) or 'content' not in section:
                            continue
                        section_heading = section.get('heading', None)
                        section_content = section.get('content', [])
                        if isinstance(section_content, list):
                            content_text = '\n'.join(str(c) for c in section_content if c)
                        else:
                            content_text = str(section_content) if section_content else ''
                        
                        # Create formatted content
                        if section_heading:
                            formatted_content = f"Heading: {section_heading}\n\nContent: {content_text}"
                        else:
                            formatted_content = f"Content: {content_text}"
                        
                        # Skip if content is empty
                        if not content_text.strip():
                            continue
                        
                        # Create metadata
                        metadata = {
                            'source': article_source,
                            'url': article_url,
                            'title': article_title,
                            'heading': section_heading,
                            'extraction_status': item.get('extraction_status', 'unknown')
                        }
                        doc = Document(
                            page_content=formatted_content,
                            metadata=metadata
                        )
                        documents.append(doc)
            
            logging.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logging.error(f"Error loading JSON file: {str(e)}")
            raise CustomException(e, sys) from e

    def _get_bm25_file_path(self, collection_name: str) -> str:
        """Generate file path for BM25 retriever storage"""
        return f"bm25_retrievers/{collection_name}_bm25.pkl"

    def _save_bm25_retriever(self, collection_name: str, bm25_retriever: BM25Retriever) -> bool:
        """Save BM25 retriever to disk"""
        try:
            os.makedirs("bm25_retrievers", exist_ok=True)
            file_path = self._get_bm25_file_path(collection_name)
            
            with open(file_path, 'wb') as f:
                pickle.dump(bm25_retriever, f)
            
            logging.info(f"BM25 retriever saved to {file_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving BM25 retriever: {str(e)}")
            return False

    def _load_bm25_retriever(self, collection_name: str) -> Optional[BM25Retriever]:
        """Load BM25 retriever from disk"""
        try:
            file_path = self._get_bm25_file_path(collection_name)
            
            if not os.path.exists(file_path):
                logging.warning(f"BM25 file {file_path} not found")
                return None
            
            with open(file_path, 'rb') as f:
                bm25_retriever = pickle.load(f)
            
            logging.info(f"BM25 retriever loaded from {file_path}")
            return bm25_retriever
        except Exception as e:
            logging.error(f"Error loading BM25 retriever: {str(e)}")
            return None

    def create_bm25_retriever(self, documents: List[Document], collection_name: str) -> BM25Retriever:
        """Create and save BM25 retriever from documents"""
        try:
            logging.info(f"Creating BM25 retriever for {len(documents)} documents")
            bm25_retriever = BM25Retriever.from_documents(documents)
            self._save_bm25_retriever(collection_name, bm25_retriever)
            logging.info("BM25 retriever created and saved successfully")
            return bm25_retriever
        except Exception as e:
            logging.error(f"Error creating BM25 retriever: {str(e)}")
            raise CustomException(e, sys) from e

    def setup_retrievers(self, collection_name: str, documents: List[Document] = None) -> bool:
        """Setup both vector and BM25 retrievers"""
        try:
            if self._collection_exists(collection_name):
                self.vector_retriever = Qdrant(
                    client=self.client,
                    collection_name=collection_name,
                    embeddings=self.embeddings
                ).as_retriever(search_kwargs={'k': top_collection_search})
                logging.info("Vector retriever setup completed")
            else:
                logging.warning(f"Collection {collection_name} does not exist for vector retriever")
                return False
            
            self.bm25_retriever = self._load_bm25_retriever(collection_name)
            
            if self.bm25_retriever is None and documents:
                logging.info("Creating new BM25 retriever from documents")
                self.bm25_retriever = self.create_bm25_retriever(documents, collection_name)
            elif self.bm25_retriever is None:
                logging.warning("No BM25 retriever found and no documents provided to create one")
                return False
            
            # Setup ensemble retriever (combines both)
            if self.vector_retriever and self.bm25_retriever:
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[self.vector_retriever, self.bm25_retriever],
                    weights=[0.6, 0.4]
                )
                logging.info("Ensemble retriever setup completed")
            
            return True
            
        except Exception as e:
            logging.error(f"Error setting up retrievers: {str(e)}")
            raise CustomException(e, sys) from e

    async def StoreInMemory(self, collection_name: str, file_path: str, chunk_size: int = 2000, chunk_overlap: int = 100) -> bool:
        """
        Store the JSON file data in the vector database and create BM25 retriever
        """
        try:
            logging.info(f"Storing JSON data from {file_path}")
            
            # Ensure embeddings are working before proceeding
            if self.embeddings is None:
                self._initialize_embeddings()
            
            # Load JSON document
            documents = self.load_json_file(file_path)
            
            if not documents:
                logging.warning("No documents found to store")
                return False
            
            logging.info(f"Processing {len(documents)} sections for collection {collection_name}")
            
            self.create_collection(collection_name)
            
            # Store documents, only split if they exceed chunk_size
            texts_to_store = []
            split_count = 0
            
            for doc in documents:
                if len(doc.page_content) > chunk_size:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    split_docs = text_splitter.split_documents([doc])
                    texts_to_store.extend(split_docs)
                    split_count += len(split_docs) - 1  
                    logging.info(f"Split large section into {len(split_docs)} chunks")
                else:
                    texts_to_store.append(doc)
            
            # Store in Qdrant with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    qdrant = Qdrant.from_documents(
                        texts_to_store,
                        self.embeddings,
                        url=self.qdrant_url,
                        collection_name=collection_name,
                        prefer_grpc=False
                    )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logging.warning(f"Attempt {attempt + 1} to store documents failed: {str(e)}, retrying...")
                        time.sleep(2 ** attempt)
                        self._initialize_embeddings()  # Reinitialize embeddings
                    else:
                        raise e
            
            # Create BM25 retriever from the same documents
            self.create_bm25_retriever(texts_to_store, collection_name)
            self.setup_retrievers(collection_name, texts_to_store)
            
            logging.info(f"Successfully stored {len(texts_to_store)} documents in collection {collection_name}")
            logging.info(f"Original sections: {len(documents)}, After splitting: {len(texts_to_store)}, Split operations: {split_count}")
            return True
            
        except Exception as e:
            logging.error(f"Error in JSON storing: {str(e)}")
            raise CustomException(e, sys) from e

    def search_in_collection(self, query: str, collection_name: str, k: int = top_collection_search) -> List:
        """Search in the collection using vector similarity"""
        try:
            # Handle query format - extract string content if it's a dict or object
            if isinstance(query, dict):
                query_str = query.get("content", str(query))
            elif hasattr(query, 'content'):
                query_str = query.content
            else:
                query_str = str(query)
                
            if not self._collection_exists(collection_name=collection_name):
                logging.warning(f"Collection {collection_name} does not exist")
                return []
                
            logging.info(f"Search in collection using vector similarity with query: {query_str}")
            
            # Ensure embeddings are working
            if self.embeddings is None:
                self._initialize_embeddings()
                
            db = Qdrant(
                client=self.client,
                collection_name=collection_name,
                embeddings=self.embeddings
            )
            docs = db.similarity_search_with_score(query=query_str, k=k)
            logging.info("Relevant docs found with vector similarity score")
            return docs
            
        except Exception as e:
            logging.error(f"Error in similarity search {str(e)}") 
            raise CustomException(e, sys) from e

    def bm25_search(self, query: str, collection_name: str, k: int = top_collection_search) -> List[Document]:
        """Search using BM25 keyword retriever"""
        try:
            # Handle query format - extract string content if it's a dict or object
            if isinstance(query, dict):
                query_str = query.get("content", str(query))
            elif hasattr(query, 'content'):
                query_str = query.content
            else:
                query_str = str(query)
                
            if self.bm25_retriever is None:
                self.bm25_retriever = self._load_bm25_retriever(collection_name)
                
                if self.bm25_retriever is None:
                    logging.warning(f"BM25 retriever not found for collection {collection_name}")
                    return []
            
            logging.info(f"Search using BM25 keyword retriever with query: {query_str}")
            self.bm25_retriever.k = k
            docs = self.bm25_retriever.get_relevant_documents(query_str)
            logging.info(f"Found {len(docs)} documents with BM25 search")
            return docs
            
        except Exception as e:
            logging.error(f"Error in BM25 search: {str(e)}")
            raise CustomException(e, sys) from e

    def hybrid_search(self, query: str, collection_name: str, k: int = top_collection_search) -> List[Document]:
        """Search using ensemble retriever (hybrid: vector + BM25)"""
        try:
            # Handle query format - extract string content if it's a dict or object
            if isinstance(query, dict):
                query_str = query.get("content", str(query))
            elif hasattr(query, 'content'):
                query_str = query.content
            else:
                query_str = str(query)
                
            logging.info(f"Hybrid search with query: {query_str}")
            
            if self.ensemble_retriever is None:
                success = self.setup_retrievers(collection_name)
                if not success:
                    logging.warning("Could not setup ensemble retriever, falling back to BM25 only")
                    return self.bm25_search(query_str, collection_name, k)
            
            logging.info("Search using hybrid retriever (vector + BM25)")
            self.ensemble_retriever.retrievers[0].search_kwargs = {'k': k} 
            if hasattr(self.ensemble_retriever.retrievers[1], 'k'):
                self.ensemble_retriever.retrievers[1].k = k 
            
            docs = self.ensemble_retriever.get_relevant_documents(query_str)
            logging.info(f"Found {len(docs)} documents with hybrid search")
            return docs
            
        except Exception as e:
            logging.error(f"Error in hybrid search: {str(e)}")
            # Fallback to BM25 search if hybrid fails
            logging.info("Falling back to BM25 search")
            return self.bm25_search(query_str, collection_name, k)

    def search_with_method(self, query: str, collection_name: str, method: str = "hybrid", k: int = top_collection_search) -> Union[List, List[Document]]:
        if method == "vector":
            return self.search_in_collection(query, collection_name, k)
        elif method == "bm25":
            return self.bm25_search(query, collection_name, k)
        elif method == "hybrid":
            return self.hybrid_search(query, collection_name, k)
        else:
            raise ValueError(f"Invalid search method: {method}. Use 'vector', 'bm25', or 'hybrid'")

memory = DataStore()  

if __name__ == "__main__":
    import asyncio
    
    async def main():
        memory = DataStore()
        collection_name = "health_articles_collection"
        
        # Uncomment to store data first
        file_path = r"alldata\gut_health_raw_data.json" 
        success = await memory.StoreInMemory(collection_name, file_path)

        query = "What is gut microbiome?"
        
        # Test all three search methods
        print(f"Search results for: '{query}'\n")
        
        # Vector search
        print("=== VECTOR SEARCH ===")
        try:
            vector_results = memory.search_in_collection(query, collection_name, k=3)
            for i, (doc, score) in enumerate(vector_results):
                print(f"\nVector Result {i+1} (Score: {score:.4f}):")
                print(f"Content: {doc.page_content[:200]}...")
                print(f"Source: {doc.metadata.get('source', 'unknown')}")
        except Exception as e:
            print(f"Vector search failed: {e}")
        
        # BM25 search
        print("\n=== BM25 KEYWORD SEARCH ===")
        try:
            bm25_results = memory.bm25_search(query, collection_name, k=3)
            for i, doc in enumerate(bm25_results):
                print(f"\nBM25 Result {i+1}:")
                print(f"Content: {doc.page_content[:200]}...")
                print(f"Source: {doc.metadata.get('source', 'unknown')}")
        except Exception as e:
            print(f"BM25 search failed: {e}")
        
        # Hybrid search
        print("\n=== HYBRID SEARCH ===")
        try:
            hybrid_results = memory.hybrid_search(query, collection_name, k=3)
            for i, doc in enumerate(hybrid_results):
                print(f"\nHybrid Result {i+1}:")
                print(f"Content: {doc.page_content[:200]}...")
                print(f"Source: {doc.metadata.get('source', 'unknown')}")
        except Exception as e:
            print(f"Hybrid search failed: {e}")
    
    asyncio.run(main())