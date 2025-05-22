import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import json
import numpy as np
from datetime import datetime
from collections import OrderedDict
import hashlib


# Load environment variables
load_dotenv()

# Directory paths
DATA_DIR = Path("data")
VECTOR_DB_DIR = DATA_DIR / "vector_db"
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class LRUCache:
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        )
        self.cache_file = CACHE_DIR / "query_cache.json"
        self._load_cache()

    def _load_cache(self):
        """Load cache from file."""
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
                self.cache = OrderedDict(data)

    def _save_cache(self):
        """Save cache to file."""
        with open(self.cache_file, 'w') as f:
            json.dump(dict(self.cache), f, indent=2)

    def _compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def get(self, query: str) -> Tuple[Optional[Dict], float]:
        """Get cached result if query is semantically similar."""
        if not self.cache:
            return None, 0.0

        query_embedding = self.embeddings.embed_query(query)
        best_similarity = 0.0
        best_result = None

        for cached_query, cached_data in self.cache.items():
            cached_embedding = self.embeddings.embed_query(cached_query)
            similarity = self._compute_similarity(query_embedding, cached_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_result = cached_data

        # Return result if similarity is above threshold
        if best_similarity >= 0.85:  # Adjust threshold as needed
            return best_result, best_similarity
        return None, 0.0

    def set(self, query: str, result: Dict):
        """Cache a query and its result."""
        # Remove oldest item if cache is full
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)

        self.cache[query] = {
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
        self._save_cache()

class DeepgramChat:
    def __init__(self):
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        )
        
        # Load vector stores
        self.docs_store = FAISS.load_local(
            str(VECTOR_DB_DIR / "docs_store"),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.sdk_store = FAISS.load_local(
            str(VECTOR_DB_DIR / "sdk_store"),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=os.getenv("LLM_MODEL", "gpt-4-turbo-preview"),
            temperature=0.7
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize LRU cache
        self.cache = LRUCache(capacity=100)
        
        # Create custom prompt template
        template = """You are a helpful AI assistant that specializes in Deepgram's APIs and SDKs. 
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Chat History: {chat_history}
        
        Human: {question}
        Assistant:"""
        
        self.prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=template
        )
        
        # Initialize retrieval chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.docs_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt}
        )
    
    def get_answer(self, question: str, force_refresh: bool = False) -> Dict:
        """Get answer for a question using the QA chain with semantic caching."""
        # Check cache first if not forcing refresh
        if not force_refresh:
            cached_result, similarity = self.cache.get(question)
            if cached_result:
                print(f"Retrieved answer from cache (similarity: {similarity:.2f})")
                return {
                    **cached_result['result'],
                    'metadata': {
                        'source': 'cache',
                        'similarity_score': similarity
                    }
                }
        
        # Get answer from QA chain
        result = self.qa_chain({"question": question})
        
        # Get relevant documents
        docs = self.docs_store.similarity_search(question, k=3)
        sdk_docs = self.sdk_store.similarity_search(question, k=2)
        
        # Format sources
        sources = []
        for doc in docs + sdk_docs:
            if "source_url" in doc.metadata:
                sources.append({
                    "type": "documentation",
                    "url": doc.metadata["source_url"],
                    "title": doc.metadata.get("title", "Documentation")
                })
            elif "source_file" in doc.metadata:
                sources.append({
                    "type": "sdk_code",
                    "file": doc.metadata["source_file"]
                })
        
        final_result = {
            "answer": result["answer"],
            "sources": sources,
            'metadata': {
                'source': 'llm'
            }
        }
        
        # Cache the result
        self.cache.set(question, final_result)
        
        return final_result


def main():
    print("Initializing Deepgram Chat...")
    chat = DeepgramChat()
    
    print("\nWelcome to Deepgram Chat! Ask me anything about Deepgram's APIs and SDKs.")
    print("Type 'exit' to quit, 'refresh' to force a fresh response.\n")
    
    while True:
        question = input("You: ").strip()
        
        if question.lower() == 'exit':
            break
        
        force_refresh = question.lower() == 'refresh'
        if force_refresh:
            question = input("Enter your question: ").strip()
        
        try:
            result = chat.get_answer(question, force_refresh)
            
            print("\nAssistant:", result["answer"])
            print(f"\nSource: {result['metadata']['source']}")
            if 'similarity_score' in result['metadata']:
                print(f"Cache similarity: {result['metadata']['similarity_score']:.2f}")
            
            if result["sources"]:
                print("\nSources:")
                for source in result["sources"]:
                    if source["type"] == "documentation":
                        print(f"- {source['title']}: {source['url']}")
                    else:
                        print(f"- SDK Code: {source['file']}")
            
            print()
            
        except Exception as e:
            print(f"\nError: {str(e)}\n")

if __name__ == "__main__":
    main() 