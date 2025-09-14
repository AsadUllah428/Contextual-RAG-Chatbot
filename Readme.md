#  DocuMind AI - RAG Document Q&A System

A simple yet powerful document Q&A system using Azure OpenAI and Qdrant vector database. Ask questions about your documents and get AI-powered answers with source citations.

##  Features

- ** Azure OpenAI Integration** - GPT-4 for responses, text-embedding-ada-002 for search
- ** Intelligent Search** - Query optimization and semantic document retrieval
- ** PDF Support** - Upload and process PDF documents
- ** Chat Interface** - Streamlit-powered conversational UI
- ** Source Citations** - Clear references to source documents
- ** Rate Limiting** - Built-in API protection

##  Quick Start

### 1. Prerequisites

- **Python 3.8+**
- **Azure OpenAI Service** with deployed models:
  - `gpt-4` (or `gpt-35-turbo`)
  - `text-embedding-ada-002`
- **Qdrant Vector Database** (assumed to be running and populated)

### 2. Installation

```bash
# Clone or download the files
# app.py, requirements.txt, README.md

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Edit `app.py` and update these configuration values:

```python
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = "your_azure_openai_api_key_here"
AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com"
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4"

# Qdrant Configuration
QDRANT_HOST = "localhost"  # or your Qdrant host
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "documents"  # your collection name
```

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

##  How to Get Azure OpenAI Credentials

1. **Create Azure OpenAI Resource**
   - Go to [Azure Portal](https://portal.azure.com)
   - Create a new "Azure OpenAI" resource
   - Choose your region and pricing tier

2. **Deploy Models**
   - Go to [Azure OpenAI Studio](https://oai.azure.com/)
   - Navigate to "Deployments"
   - Deploy these models:
     - **GPT-4** (for chat responses)
     - **text-embedding-ada-002** (for document embeddings)

3. **Get Your Credentials**
   - In Azure Portal, go to your OpenAI resource
   - Copy the **API Key** from "Keys and Endpoint"
   - Copy the **Endpoint URL**

##  Qdrant Setup

This app assumes you already have a Qdrant vector database with documents. If you need to set up Qdrant:

### Option A: Local Qdrant (Development)
```bash
# Using Docker
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

### Option B: Qdrant Cloud (Production)
1. Sign up at [Qdrant Cloud](https://cloud.qdrant.io/)
2. Create a cluster
3. Update the configuration in `app.py`:

```python
QDRANT_HOST = "your-cluster-url.qdrant.tech"
QDRANT_PORT = 443  # or 6333
# Add API key if using cloud
```

##  Usage

1. **Start the App**: Run `streamlit run app.py`

2. **Ask Questions**: Type your question in the chat input at the bottom

3. **View Results**: 
   - Get AI-generated answers based on your documents
   - See query optimization details
   - Check document sources and citations

4. **Optional**: Upload new PDFs to add them to your vector database

##  Configuration Options

You can modify these settings in `app.py`:

```python
# App Settings
CHUNK_SIZE = 1000              # Document chunk size
CHUNK_OVERLAP = 200            # Overlap between chunks
SIMILARITY_THRESHOLD = 0.85    # Minimum similarity for search
MAX_DOCS_PER_QUERY = 5         # Max documents to retrieve

# Model Settings
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4"           # or "gpt-35-turbo" for faster responses
```

##  Troubleshooting

### Common Issues

** "Azure OpenAI not configured"**
- Update `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` in `app.py`
- Ensure your models are deployed in Azure OpenAI Studio

** "Qdrant connection failed"**
- Check if Qdrant is running: `curl http://localhost:6333/health`
- Verify `QDRANT_HOST` and `QDRANT_PORT` settings
- Ensure your collection exists and has the correct name

** "No relevant documents found"**
- Lower the `SIMILARITY_THRESHOLD` (try 0.7 instead of 0.85)
- Try rephrasing your question
- Check if your vector database has documents

** Rate limit errors**
- Wait 60 seconds between heavy operations
- The app has built-in rate limiting to prevent issues

### Testing Your Setup

1. **Test Azure OpenAI**:
```bash
curl -H "api-key: YOUR_API_KEY" \
     "YOUR_ENDPOINT/openai/deployments/gpt-4/chat/completions?api-version=2024-02-01" \
     -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":10}'
```

2. **Test Qdrant**:
```bash
curl http://localhost:6333/health
curl http://localhost:6333/collections
```

##  Features Explained

### Query Optimization
The app automatically rewrites your questions to improve document retrieval:
- **Original**: "How to install software?"
- **Optimized**: "software installation process setup guide instructions download configure"

### Smart Retrieval
- Uses semantic similarity search to find relevant document chunks
- Adjustable similarity threshold for precision vs. recall
- Fallback search with lower threshold if no results found

### Source Attribution
Every answer includes:
- Document name and page numbers
- Exact text chunks used for the answer
- Expandable source section for verification

##  Security Notes

- **Never commit API keys** to version control
- Consider using environment variables for production:
```python
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "fallback_key")
```
- Monitor your Azure OpenAI usage in the Azure portal
- Use proper network security for production deployments

##  Performance Tips

1. **Optimize Chunk Size**: Smaller chunks (500-800) for specific answers, larger chunks (1000-1500) for comprehensive responses

2. **Adjust Similarity Threshold**: 
   - Higher (0.9): More precise, fewer results
   - Lower (0.7): More inclusive, may include less relevant content

3. **Model Selection**:
   - `gpt-4`: Best quality, slower, more expensive
   - `gpt-35-turbo`: Faster, cheaper, good quality

##  Support

- **Azure Issues**: Check [Azure OpenAI Documentation](https://docs.microsoft.com/azure/cognitive-services/openai/)
- **Qdrant Issues**: Visit [Qdrant Documentation](https://qdrant.tech/documentation/)
- **Streamlit Issues**: See [Streamlit Docs](https://docs.streamlit.io/)

##  File Structure

```
your-project/
├── app.py              # Main application
├── requirements.txt    # Python dependencies  
└── README.md          # This file
```

That's it! Three simple files for a powerful document Q&A system. 

---

