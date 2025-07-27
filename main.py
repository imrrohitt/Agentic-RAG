"""
Agentic RAG System - Core Implementation
========================================

This module implements a high-level, open-source agentic Retrieval-Augmented Generation (RAG) 
system that integrates LangChain for agentic workflows, LlamaIndex for advanced indexing and 
retrieval, the Gemini API as the primary LLM, and FAISS for efficient vector storage.

Features:
- Multi-step reasoning with master and sub-agents
- Hybrid search (semantic + keyword)
- Multimodal data ingestion (text and PDFs)
- Memory module for contextual continuity
- Production-ready architecture

Author: Open Source Community
License: MIT
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import pickle
from datetime import datetime

# Core dependencies
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# LangChain imports
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    ServiceContext, 
    Document,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

# PDF processing
import PyPDF2
from io import BytesIO

# Search and NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeminiLLM(LLM):
    """Custom LangChain LLM wrapper for Google Gemini API."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        super().__init__()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
    
    @property
    def _llm_type(self) -> str:
        return "gemini"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return f"Error: Unable to generate response - {str(e)}"


class MemoryModule:
    """Manages conversation context and memory for the RAG system."""
    
    def __init__(self, max_memory_size: int = 50):
        self.max_memory_size = max_memory_size
        self.conversation_history: List[Dict[str, Any]] = []
        self.context_summary = ""
        
    def add_interaction(self, query: str, response: str, metadata: Dict[str, Any] = None):
        """Add a new interaction to memory."""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "metadata": metadata or {}
        }
        
        self.conversation_history.append(interaction)
        
        # Maintain memory size limit
        if len(self.conversation_history) > self.max_memory_size:
            self.conversation_history.pop(0)
    
    def get_recent_context(self, n: int = 5) -> str:
        """Get recent conversation context."""
        recent = self.conversation_history[-n:] if self.conversation_history else []
        context = ""
        for interaction in recent:
            context += f"Human: {interaction['query']}\nAssistant: {interaction['response']}\n\n"
        return context
    
    def summarize_context(self) -> str:
        """Generate a summary of the conversation context."""
        if not self.conversation_history:
            return ""
        
        # Simple summarization - in production, you might use an LLM for this
        topics = set()
        for interaction in self.conversation_history:
            # Extract key topics (simplified approach)
            words = re.findall(r'\b\w+\b', interaction['query'].lower())
            topics.update([w for w in words if len(w) > 4])
        
        return f"Recent discussion topics: {', '.join(list(topics)[:10])}"


class HybridSearchEngine:
    """Implements hybrid search combining semantic and keyword search."""
    
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.documents: List[str] = []
        self.tfidf_matrix = None
        
    def add_documents(self, documents: List[str]):
        """Add documents to the search index."""
        self.documents.extend(documents)
        if len(self.documents) > 0:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.documents)
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Perform keyword-based search using TF-IDF."""
        if self.tfidf_matrix is None:
            return []
        
        query_vector = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k results
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [(idx, similarities[idx]) for idx in top_indices if similarities[idx] > 0.1]
    
    def semantic_search(self, query: str, faiss_index: faiss.Index, top_k: int = 5) -> List[Tuple[int, float]]:
        """Perform semantic search using FAISS."""
        query_embedding = self.embedding_model.encode([query])
        distances, indices = faiss_index.search(query_embedding.astype('float32'), top_k)
        
        return [(int(indices[0][i]), float(distances[0][i])) for i in range(len(indices[0]))]
    
    def hybrid_search(
        self, 
        query: str, 
        faiss_index: faiss.Index, 
        top_k: int = 5, 
        semantic_weight: float = 0.7
    ) -> List[Tuple[int, float, str]]:
        """Combine semantic and keyword search results."""
        semantic_results = self.semantic_search(query, faiss_index, top_k)
        keyword_results = self.keyword_search(query, top_k)
        
        # Combine and re-rank results
        combined_scores = {}
        
        # Add semantic scores
        for idx, score in semantic_results:
            combined_scores[idx] = semantic_weight * (1 / (1 + score))  # Convert distance to similarity
        
        # Add keyword scores
        for idx, score in keyword_results:
            if idx in combined_scores:
                combined_scores[idx] += (1 - semantic_weight) * score
            else:
                combined_scores[idx] = (1 - semantic_weight) * score
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return with document content
        results = []
        for idx, score in sorted_results[:top_k]:
            if idx < len(self.documents):
                results.append((idx, score, self.documents[idx]))
        
        return results


class RetrievalAgent:
    """Sub-agent responsible for document retrieval and search operations."""
    
    def __init__(self, hybrid_search: HybridSearchEngine, faiss_index: faiss.Index):
        self.hybrid_search = hybrid_search
        self.faiss_index = faiss_index
        
    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a given query."""
        try:
            results = self.hybrid_search.hybrid_search(query, self.faiss_index, top_k)
            
            retrieved_docs = []
            for idx, score, content in results:
                retrieved_docs.append({
                    "index": idx,
                    "score": score,
                    "content": content[:500] + "..." if len(content) > 500 else content,
                    "full_content": content
                })
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query[:50]}...")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            return []


class GenerationAgent:
    """Sub-agent responsible for response generation using retrieved context."""
    
    def __init__(self, llm: GeminiLLM):
        self.llm = llm
        
    def generate_response(
        self, 
        query: str, 
        retrieved_docs: List[Dict[str, Any]], 
        conversation_context: str = ""
    ) -> str:
        """Generate a response using retrieved documents and context."""
        try:
            # Prepare context from retrieved documents
            context = "\n\n".join([
                f"Document {i+1} (Score: {doc['score']:.3f}):\n{doc['content']}"
                for i, doc in enumerate(retrieved_docs)
            ])
            
            # Create prompt
            prompt = f"""
You are an intelligent assistant with access to relevant documents. Use the provided context to answer the user's question accurately and comprehensively.

Conversation Context:
{conversation_context}

Retrieved Context:
{context}

User Question: {query}

Instructions:
1. Base your answer primarily on the retrieved documents
2. If the context doesn't contain sufficient information, clearly state this
3. Provide specific references to the source material when possible
4. Be concise but thorough
5. If asked about something not in the context, explain what information is available instead

Answer:
"""
            
            response = self.llm._call(prompt)
            logger.info(f"Generated response for query: {query[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"Error in response generation: {e}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"


class MasterAgent:
    """Master agent that orchestrates the overall RAG workflow."""
    
    def __init__(
        self, 
        retrieval_agent: RetrievalAgent, 
        generation_agent: GenerationAgent,
        memory_module: MemoryModule,
        llm: GeminiLLM
    ):
        self.retrieval_agent = retrieval_agent
        self.generation_agent = generation_agent
        self.memory_module = memory_module
        self.llm = llm
        
        # Initialize LangChain memory
        self.langchain_memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=5,  # Keep last 5 exchanges
            return_messages=True
        )
        
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the user query to determine the best approach."""
        analysis_prompt = f"""
Analyze the following user query and determine:
1. Query type (factual, analytical, conversational, etc.)
2. Required information retrieval approach
3. Complexity level (simple, moderate, complex)
4. Whether it requires multi-step reasoning

Query: {query}

Provide analysis in JSON format:
{{
    "query_type": "type",
    "retrieval_approach": "approach",
    "complexity": "level",
    "multi_step": boolean,
    "key_concepts": ["concept1", "concept2"]
}}
"""
        
        try:
            analysis_response = self.llm._call(analysis_prompt)
            # Parse JSON response (with error handling)
            try:
                analysis = json.loads(analysis_response)
            except json.JSONDecodeError:
                # Fallback analysis
                analysis = {
                    "query_type": "factual",
                    "retrieval_approach": "hybrid",
                    "complexity": "moderate",
                    "multi_step": False,
                    "key_concepts": [query.split()[:3]]
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            return {
                "query_type": "factual",
                "retrieval_approach": "hybrid",
                "complexity": "moderate",
                "multi_step": False,
                "key_concepts": []
            }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Main method to process user queries through the agentic workflow."""
        logger.info(f"Processing query: {query[:100]}...")
        
        try:
            # Step 1: Analyze query
            analysis = self.analyze_query(query)
            
            # Step 2: Get conversation context
            context = self.memory_module.get_recent_context()
            
            # Step 3: Retrieve relevant documents
            retrieved_docs = self.retrieval_agent.retrieve_documents(
                query, 
                top_k=10 if analysis.get("complexity") == "complex" else 5
            )
            
            # Step 4: Generate response
            response = self.generation_agent.generate_response(
                query, retrieved_docs, context
            )
            
            # Step 5: Store in memory
            self.memory_module.add_interaction(
                query, 
                response, 
                {"analysis": analysis, "retrieved_docs_count": len(retrieved_docs)}
            )
            
            # Step 6: Update LangChain memory
            self.langchain_memory.chat_memory.add_user_message(query)
            self.langchain_memory.chat_memory.add_ai_message(response)
            
            return {
                "query": query,
                "response": response,
                "analysis": analysis,
                "retrieved_documents": len(retrieved_docs),
                "sources": [{"index": doc["index"], "score": doc["score"]} for doc in retrieved_docs[:3]]
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_response = f"I apologize, but I encountered an error while processing your query: {str(e)}"
            
            return {
                "query": query,
                "response": error_response,
                "analysis": {},
                "retrieved_documents": 0,
                "sources": [],
                "error": str(e)
            }


class AgenticRAGSystem:
    """Main RAG system class that coordinates all components."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_path = Path(config.get("data_path", "./data"))
        self.storage_path = Path(config.get("storage_path", "./storage"))
        
        # Ensure directories exist
        self.data_path.mkdir(exist_ok=True)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.embedding_model = None
        self.faiss_index = None
        self.hybrid_search = None
        self.retrieval_agent = None
        self.generation_agent = None
        self.memory_module = None
        self.master_agent = None
        self.documents = []
        
        logger.info("Agentic RAG System initialized")
    
    def initialize_models(self):
        """Initialize all ML models and components."""
        logger.info("Initializing models...")
        
        # Initialize embedding model
        model_name = self.config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize Gemini LLM
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        gemini_llm = GeminiLLM(api_key)
        
        # Initialize hybrid search
        self.hybrid_search = HybridSearchEngine(self.embedding_model)
        
        # Initialize memory module
        self.memory_module = MemoryModule(max_memory_size=50)
        
        logger.info("Models initialized successfully")
        
        return gemini_llm
    
    def load_documents(self) -> List[Document]:
        """Load and process documents from the data directory."""
        logger.info("Loading documents...")
        
        documents = []
        
        # Load text files
        for file_path in self.data_path.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(Document(text=content, metadata={"source": str(file_path)}))
                    logger.info(f"Loaded text file: {file_path}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        # Load PDF files
        for file_path in self.data_path.glob("*.pdf"):
            try:
                content = self.extract_pdf_text(file_path)
                if content:
                    documents.append(Document(text=content, metadata={"source": str(file_path)}))
                    logger.info(f"Loaded PDF file: {file_path}")
            except Exception as e:
                logger.error(f"Error loading PDF {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF files."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            logger.error(f"Error extracting PDF text from {pdf_path}: {e}")
            return ""
    
    def create_vector_index(self, documents: List[Document]):
        """Create FAISS vector index from documents."""
        logger.info("Creating vector index...")
        
        # Parse documents into nodes
        node_parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)
        nodes = node_parser.get_nodes_from_documents(documents)
        
        # Extract text content for embedding
        texts = [node.text for node in nodes]
        self.documents = texts  # Store for hybrid search
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings.astype('float32'))
        self.faiss_index.add(embeddings.astype('float32'))
        
        # Add documents to hybrid search
        self.hybrid_search.add_documents(texts)
        
        logger.info(f"Created vector index with {len(texts)} chunks")
    
    def save_index(self):
        """Save the FAISS index and related data."""
        try:
            # Save FAISS index
            faiss.write_index(self.faiss_index, str(self.storage_path / "faiss_index.bin"))
            
            # Save documents and metadata
            with open(self.storage_path / "documents.pkl", 'wb') as f:
                pickle.dump(self.documents, f)
            
            # Save hybrid search data
            with open(self.storage_path / "tfidf_vectorizer.pkl", 'wb') as f:
                pickle.dump(self.hybrid_search.tfidf_vectorizer, f)
            
            with open(self.storage_path / "tfidf_matrix.pkl", 'wb') as f:
                pickle.dump(self.hybrid_search.tfidf_matrix, f)
            
            logger.info("Index saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def load_index(self) -> bool:
        """Load existing FAISS index and related data."""
        try:
            # Check if index files exist
            index_path = self.storage_path / "faiss_index.bin"
            docs_path = self.storage_path / "documents.pkl"
            
            if not (index_path.exists() and docs_path.exists()):
                return False
            
            # Load FAISS index
            self.faiss_index = faiss.read_index(str(index_path))
            
            # Load documents
            with open(docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            
            # Restore hybrid search
            self.hybrid_search.documents = self.documents
            
            # Load TF-IDF data if available
            tfidf_vec_path = self.storage_path / "tfidf_vectorizer.pkl"
            tfidf_mat_path = self.storage_path / "tfidf_matrix.pkl"
            
            if tfidf_vec_path.exists() and tfidf_mat_path.exists():
                with open(tfidf_vec_path, 'rb') as f:
                    self.hybrid_search.tfidf_vectorizer = pickle.load(f)
                
                with open(tfidf_mat_path, 'rb') as f:
                    self.hybrid_search.tfidf_matrix = pickle.load(f)
            else:
                # Recreate TF-IDF if not saved
                if self.documents:
                    self.hybrid_search.tfidf_matrix = self.hybrid_search.tfidf_vectorizer.fit_transform(self.documents)
            
            logger.info(f"Loaded existing index with {len(self.documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def setup_system(self):
        """Complete system setup including model initialization and data loading."""
        # Initialize models
        gemini_llm = self.initialize_models()
        
        # Try to load existing index
        if not self.load_index():
            # Load and process documents
            documents = self.load_documents()
            
            if not documents:
                logger.warning("No documents found. Creating empty index.")
                # Create empty index for now
                self.faiss_index = faiss.IndexFlatIP(384)  # Default dimension for MiniLM
                self.documents = []
            else:
                # Create new index
                self.create_vector_index(documents)
                # Save the new index
                self.save_index()
        
        # Initialize agents
        self.retrieval_agent = RetrievalAgent(self.hybrid_search, self.faiss_index)
        self.generation_agent = GenerationAgent(gemini_llm)
        self.master_agent = MasterAgent(
            self.retrieval_agent,
            self.generation_agent,
            self.memory_module,
            gemini_llm
        )
        
        logger.info("Agentic RAG System setup complete")
    
    def ingest_text(self, text: str, metadata: Dict[str, Any] = None) -> bool:
        """Ingest new text into the system."""
        try:
            # Create document
            doc = Document(text=text, metadata=metadata or {})
            
            # Parse into chunks
            node_parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)
            nodes = node_parser.get_nodes_from_documents([doc])
            
            # Get text chunks
            new_texts = [node.text for node in nodes]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(new_texts)
            
            # Add to FAISS index
            faiss.normalize_L2(embeddings.astype('float32'))
            self.faiss_index.add(embeddings.astype('float32'))
            
            # Add to documents and hybrid search
            self.documents.extend(new_texts)
            self.hybrid_search.add_documents(new_texts)
            
            # Save updated index
            self.save_index()
            
            logger.info(f"Ingested text with {len(new_texts)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting text: {e}")
            return False
    
    def ingest_pdf(self, pdf_path: str) -> bool:
        """Ingest PDF file into the system."""
        try:
            text = self.extract_pdf_text(Path(pdf_path))
            if text:
                return self.ingest_text(text, {"source": pdf_path, "type": "pdf"})
            return False
        
        except Exception as e:
            logger.error(f"Error ingesting PDF: {e}")
            return False
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process a query through the agentic RAG system."""
        if not self.master_agent:
            raise RuntimeError("System not properly initialized. Call setup_system() first.")
        
        return self.master_agent.process_query(question)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics and health information."""
        return {
            "total_documents": len(self.documents),
            "faiss_index_size": self.faiss_index.ntotal if self.faiss_index else 0,
            "memory_interactions": len(self.memory_module.conversation_history) if self.memory_module else 0,
            "embedding_model": self.config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            "system_status": "healthy" if self.master_agent else "not_initialized"
        }


def create_sample_data():
    """Create sample data for testing purposes."""
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # Sample text content
    sample_texts = [
        {
            "filename": "ai_overview.txt",
            "content": """
Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines 
that can perform tasks typically requiring human intelligence. These tasks include learning, reasoning, 
problem-solving, perception, and language understanding.

Machine Learning is a subset of AI that enables computers to learn and improve from experience without 
being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions 
or decisions.

Deep Learning is a subset of machine learning that uses neural networks with multiple layers to model 
and understand complex patterns in data. It has been particularly successful in areas like image 
recognition, natural language processing, and game playing.

Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers 
and humans through natural language. It involves developing algorithms and models that can understand, 
interpret, and generate human language.
"""
        },
        {
            "filename": "rag_systems.txt", 
            "content": """
Retrieval-Augmented Generation (RAG) is an AI framework that combines information retrieval with 
text generation. RAG systems first retrieve relevant documents from a knowledge base, then use 
this information to generate more accurate and contextually relevant responses.

The key components of a RAG system include:
1. Document Ingestion: Processing and storing documents in a searchable format
2. Embedding Generation: Converting text into vector representations
3. Vector Storage: Storing embeddings in a database like FAISS or Pinecone
4. Retrieval: Finding relevant documents based on query similarity
5. Generation: Using retrieved context to generate responses with an LLM

Benefits of RAG systems:
- Improved accuracy by grounding responses in factual information
- Ability to work with domain-specific knowledge
- Reduced hallucination compared to pure generative models
- Scalability to large document collections
- Real-time updates to knowledge base without retraining
"""
        },
        {
            "filename": "python_programming.txt",
            "content": """
Python is a high-level, interpreted programming language known for its simplicity and readability. 
It was created by Guido van Rossum and first released in 1991. Python's design philosophy emphasizes 
code readability with its notable use of significant whitespace.

Key features of Python:
- Easy to learn and use
- Extensive standard library
- Large ecosystem of third-party packages
- Cross-platform compatibility
- Support for multiple programming paradigms

Python is widely used in:
- Web development (Django, Flask)
- Data science and analytics (pandas, NumPy, matplotlib)
- Machine learning and AI (scikit-learn, TensorFlow, PyTorch)
- Automation and scripting
- Scientific computing
- Desktop applications

Popular Python frameworks and libraries:
- FastAPI: Modern web framework for building APIs
- LangChain: Framework for developing LLM applications
- Streamlit: Framework for creating data applications
- Jupyter: Interactive computing environment
"""
        }
    ]
    
    # Write sample files
    for sample in sample_texts:
        file_path = data_dir / sample["filename"]
        if not file_path.exists():
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(sample["content"])
            logger.info(f"Created sample file: {file_path}")


# Main execution
if __name__ == "__main__":
    # Configuration
    config = {
        "data_path": "./data",
        "storage_path": "./storage", 
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
    }
    
    # Create sample data
    create_sample_data()
    
    # Initialize and setup system
    rag_system = AgenticRAGSystem(config)
    rag_system.setup_system()
    
    # Test queries
    test_queries = [
        "What is artificial intelligence?",
        "How do RAG systems work?",
        "What are the benefits of using Python for machine learning?",
        "Can you explain the key components of a RAG system?"
    ]
    
    print("=" * 80)
    print("AGENTIC RAG SYSTEM - DEMO")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\n🤔 Query: {query}")
        print("-" * 40)
        
        result = rag_system.query(query)
        print(f"🤖 Response: {result['response']}")
        print(f"📊 Retrieved {result['retrieved_documents']} documents")
        
        if result.get('sources'):
            print("📚 Top sources:")
            for i, source in enumerate(result['sources'][:2], 1):
                print(f"   {i}. Document {source['index']} (Score: {source['score']:.3f})")
        
        print()
    
    # Display system stats
    stats = rag_system.get_system_stats()
    print("=" * 80)
    print("SYSTEM STATISTICS")
    print("=" * 80)
    for key, value in stats.items():
        print(f"{key}: {value}")
