#!/usr/bin/env python3
"""
DocuMind AI - Enhanced RAG Application
A sophisticated document Q&A system with Azure OpenAI & Qdrant
"""

import os
import tempfile
import time
from datetime import datetime
from typing import List, Optional
from functools import wraps

import streamlit as st
from openai import AzureOpenAI
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# ============================================
# CONFIGURATION - Update these values
# ============================================
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = "your_azure_openai_api_key_here"
AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com"
AZURE_API_VERSION = "2024-02-01"
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4"

# Qdrant Configuration (assuming existing vector DB)
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "documents"
EMBEDDING_DIMENSION = 1536

# App Configuration
APP_NAME = "DocuMind AI"
APP_ICON = "🧠"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SIMILARITY_THRESHOLD = 0.85
MAX_DOCS_PER_QUERY = 5

# ============================================
# Streamlit Configuration
# ============================================
st.set_page_config(
    page_title=f"{APP_ICON} {APP_NAME}",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Rate Limiting & Error Handling
# ============================================
def rate_limit(max_calls: int = 10, time_window: int = 60):
    """Rate limiting decorator to prevent API overuse"""
    def decorator(func):
        calls = []
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            calls[:] = [call_time for call_time in calls if now - call_time < time_window]
            
            if len(calls) >= max_calls:
                wait_time = time_window - (now - calls[0])
                st.warning(f"⏳ Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time + 1)
                calls[:] = []
            
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

def exponential_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Exponential backoff decorator for API calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_msg = str(e).lower()
                    
                    if any(keyword in error_msg for keyword in ['429', 'rate limit', 'quota']):
                        if attempt < max_retries:
                            delay = base_delay * (2 ** attempt)
                            st.warning(f"⏳ Rate limited. Retrying in {delay:.1f}s...")
                            time.sleep(delay)
                            continue
                    
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        st.warning(f"⏳ API error. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        raise e
            
            raise last_exception
        return wrapper
    return decorator

# ============================================
# Session State Management
# ============================================
def init_session_state():
    """Initialize session state with defaults"""
    defaults = {
        "vector_store": None,
        "processed_documents": [],
        "chat_history": [],
        "api_call_count": 0,
        "last_api_reset": time.time(),
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# ============================================
# Azure OpenAI Embedder
# ============================================
class AzureOpenAIEmbedder(Embeddings):
    """Azure OpenAI embeddings with rate limiting"""
    
    def __init__(self):
        self._configure_client()

    def _configure_client(self):
        """Configure Azure OpenAI client"""
        if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
            raise ValueError("Azure OpenAI API Key and Endpoint are required")
        
        self.client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_API_VERSION
        )

    @rate_limit(max_calls=20, time_window=60)
    @exponential_backoff(max_retries=3)
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        embeddings = []
        batch_size = 16
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch
                )
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                st.warning(f"Failed to embed batch: {str(e)[:100]}...")
                for _ in batch:
                    embeddings.append([0.0] * EMBEDDING_DIMENSION)
            
            if i + batch_size < len(texts):
                time.sleep(0.5)
        
        return embeddings

    @rate_limit(max_calls=30, time_window=60)
    @exponential_backoff(max_retries=3)
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[text]
        )
        return response.data[0].embedding

# ============================================
# Qdrant Operations
# ============================================
@exponential_backoff(max_retries=3, base_delay=2)
def init_qdrant() -> Optional[QdrantClient]:
    """Initialize Qdrant client (assumes existing vector DB)"""
    try:
        client = QdrantClient(
            url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
            prefer_grpc=False,
            timeout=60
        )
        
        # Test connection
        client.get_collections()
        return client
    except Exception as e:
        st.error(f" Qdrant connection failed: {e}")
        return None

def get_vector_store(client: QdrantClient) -> Optional[QdrantVectorStore]:
    """Get existing vector store"""
    try:
        return QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION_NAME,
            embedding=AzureOpenAIEmbedder()
        )
    except Exception as e:
        st.error(f" Vector store error: {e}")
        return None

# ============================================
# PDF Processing (Optional - for new uploads)
# ============================================
def process_pdf(file) -> List:
    """Process PDF file with metadata"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        os.unlink(tmp_path)
        
        if not docs:
            st.warning(" No content extracted from PDF")
            return []
        
        # Add metadata
        for i, doc in enumerate(docs):
            doc.metadata.update({
                "source_type": "pdf",
                "file_name": file.name,
                "page_number": i + 1,
                "timestamp": datetime.utcnow().isoformat(),
            })
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = splitter.split_documents(docs)
        st.info(f" Processed {len(docs)} pages into {len(chunks)} chunks")
        
        return chunks
        
    except Exception as e:
        st.error(f" PDF processing error: {e}")
        return []

# ============================================
# AI Agents
# ============================================
@rate_limit(max_calls=10, time_window=60)
@exponential_backoff(max_retries=3)
def get_query_rewriter_agent() -> Agent:
    """Create query rewriting agent"""
    return Agent(
        name="Query Optimizer",
        model=OpenAIChat(
            id=CHAT_MODEL,
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_API_VERSION
        ),
        instructions="""You are an expert at optimizing search queries for document retrieval.

Guidelines:
- Make queries more specific and detailed
- Add relevant synonyms and related terms
- Preserve the original intent
- Focus on key concepts that would appear in documents
- Return ONLY the optimized query without explanations

Example:
Input: "How to install software?"
Output: "software installation process setup guide instructions download configure""",
        show_tool_calls=False,
        markdown=True,
    )

@rate_limit(max_calls=5, time_window=60)
@exponential_backoff(max_retries=3)
def get_rag_agent() -> Agent:
    """Create RAG response agent"""
    return Agent(
        name=f"{APP_NAME} Assistant",
        model=OpenAIChat(
            id=CHAT_MODEL,
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_API_VERSION
        ),
        instructions=f"""You are {APP_NAME}, an intelligent document analysis assistant.

Your role:
- Provide accurate answers based strictly on document context
- Cite specific details from the documents
- Organize responses clearly with proper structure
- Use markdown formatting for readability
- If information isn't in context, clearly state this
- Synthesize information from multiple sources coherently
- Be precise and avoid speculation

Always maintain a helpful, professional tone.""",
        show_tool_calls=True,
        markdown=True,
    )

# ============================================
# UI Components
# ============================================
def render_sidebar():
    """Render sidebar with optional file upload"""
    st.sidebar.header("🔧 Settings")
    
    # Optional file upload (if you want to add more documents)
    with st.sidebar.expander(" Add Documents (Optional)"):
        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            help="Add new documents to the existing vector database"
        )
    
    # Settings
    with st.sidebar.expander("⚙️ Query Settings"):
        similarity = st.slider(
            "Similarity Threshold", 
            0.0, 1.0, SIMILARITY_THRESHOLD, 0.05,
            help="Minimum similarity score for retrieved documents"
        )
        max_docs = st.slider(
            "Max Documents", 
            1, 10, MAX_DOCS_PER_QUERY, 1,
            help="Maximum number of documents to retrieve"
        )
    
    # API Usage Monitor
    with st.sidebar.expander("📊 API Usage"):
        current_time = time.time()
        if current_time - st.session_state.last_api_reset > 60:
            st.session_state.api_call_count = 0
            st.session_state.last_api_reset = current_time
        
        st.metric("API Calls (last 60s)", st.session_state.api_call_count)
        
        if st.session_state.api_call_count > 25:
            st.warning(" High API usage detected")
    
    # Document Status
    if st.session_state.processed_documents:
        st.sidebar.header("📚 Processed Documents")
        for doc in st.session_state.processed_documents:
            st.sidebar.text(f" {doc}")
    
    return uploaded_file, similarity, max_docs

def render_main_interface():
    """Render main interface"""
    st.title(f"{APP_ICON} {APP_NAME}")
    st.markdown("Ask questions about your documents using advanced AI-powered search and generation.")
    
    # Info banner
    st.info("💡 **Ready to use!** The vector database is already configured. Start asking questions below.")
    
    # Status indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
            st.success(" Azure OpenAI")
        else:
            st.error(" Azure OpenAI - Configure in code")
    
    with col2:
        if init_qdrant():
            st.success(" Qdrant Connected")
        else:
            st.error(" Qdrant Offline")
    
    with col3:
        if st.session_state.vector_store:
            st.success(" Vector DB Ready")
        else:
            st.warning(" Initializing...")

# ============================================
# Query Processing
# ============================================
def process_query(prompt: str, similarity_threshold: float, max_docs: int):
    """Process user query with RAG"""
    if not st.session_state.vector_store:
        st.warning(" Vector store not available")
        return
    
    try:
        st.session_state.api_call_count += 1
        
        # Setup retriever
        retriever = st.session_state.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": max_docs,
                "score_threshold": similarity_threshold
            }
        )
        
        # Optimize query
        with st.spinner("🔄 Optimizing query..."):
            rewriter = get_query_rewriter_agent()
            optimized = rewriter.run(prompt).content.strip()
        
        # Show optimization
        with st.expander(" Query Optimization"):
            st.write(f"**Original:** {prompt}")
            st.write(f"**Optimized:** {optimized}")
        
        # Retrieve documents
        with st.spinner(" Searching documents..."):
            docs = retriever.get_relevant_documents(optimized)
        
        if not docs:
            st.warning(f"No relevant documents found (similarity > {similarity_threshold})")
            # Try with original query and lower threshold
            st.info(" Trying with original query and lower threshold...")
            retriever_fallback = st.session_state.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": max_docs,
                    "score_threshold": max(0.5, similarity_threshold - 0.2)
                }
            )
            docs = retriever_fallback.get_relevant_documents(prompt)
        
        if not docs:
            st.error(" No relevant documents found. Try rephrasing your question.")
            return
        
        st.info(f" Found {len(docs)} relevant document chunks")
        
        # Prepare context
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("file_name", "Document")
            page = doc.metadata.get("page_number", "Unknown")
            context_parts.append(f"[Source {i}: {source}, Page {page}]\n{doc.page_content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate answer
        with st.spinner("🤖 Generating answer..."):
            rag_agent = get_rag_agent()
            full_prompt = f"""Context from documents:
{context}

Question: {prompt}
Optimized Query: {optimized}

Please provide a comprehensive answer based on the context above. Include relevant details and cite your sources clearly."""
            
            answer = rag_agent.run(full_prompt)
        
        # Display results
        st.session_state.chat_history.append({"role": "assistant", "content": answer.content})
        st.chat_message("assistant").write(answer.content)
        
        # Show sources
        with st.expander(" Document Sources"):
            for i, doc in enumerate(docs, 1):
                source_name = doc.metadata.get("file_name", "Unknown Document")
                page_num = doc.metadata.get("page_number", "?")
                
                st.write(f"** Source {i}: {source_name} (Page {page_num})**")
                # Show first 300 characters of content
                content_preview = doc.page_content[:300]
                if len(doc.page_content) > 300:
                    content_preview += "..."
                st.write(content_preview)
                
                if i < len(docs):  # Don't add divider after last item
                    st.divider()
                
    except Exception as e:
        error_msg = str(e)
        if '429' in error_msg or 'rate limit' in error_msg.lower():
            st.error(" Rate limit exceeded. Please wait before trying again.")
        else:
            st.error(f" Error processing query: {e}")

# ============================================
# Main Application
# ============================================
def main():
    """Main application entry point"""
    # Initialize session state
    init_session_state()
    
    # Connect to existing Qdrant collection
    if st.session_state.vector_store is None:
        client = init_qdrant()
        if client:
            st.session_state.vector_store = get_vector_store(client)
    
    # Render UI
    uploaded_file, similarity_threshold, max_docs = render_sidebar()
    render_main_interface()
    
    # Handle optional file upload
    if uploaded_file and uploaded_file.name not in st.session_state.processed_documents:
        with st.spinner(" Processing new document..."):
            chunks = process_pdf(uploaded_file)
            if chunks and st.session_state.vector_store:
                try:
                    st.session_state.vector_store.add_documents(chunks)
                    st.session_state.processed_documents.append(uploaded_file.name)
                    st.success(f" Added {uploaded_file.name} to vector database!")
                    st.rerun()
                except Exception as e:
                    st.error(f" Failed to add document: {e}")
    
    # Display chat history
    for message in st.session_state.chat_history:
        st.chat_message(message["role"]).write(message["content"])
    
    # Handle new queries
    if prompt := st.chat_input("Ask about your documents..."):
        # Validate configuration
        if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
            st.error(" Azure OpenAI not configured. Please update the configuration in the code.")
            return
        
        if not st.session_state.vector_store:
            st.error(" Vector database not available. Please check Qdrant connection.")
            return
        
        if st.session_state.api_call_count > 30:
            st.warning(" API rate limit approaching. Please wait.")
            return
        
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Process query
        process_query(prompt, similarity_threshold, max_docs)

if __name__ == "__main__":
    main()
