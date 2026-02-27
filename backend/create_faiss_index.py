"""
Script to create and save FAISS index from documents
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_KEY or ""

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH = os.path.join(BASE_DIR, "books")
INDEX_OUTPUT_PATH = os.path.join(BASE_DIR, "faiss_index")
SAMPLE_DOCS_MODE = True  # Set to False if you have actual PDF files

def create_faiss_index_from_documents(docs_path: str, output_path: str):
    """Create FAISS index from documents"""
    try:
        logger.info("Starting FAISS index creation...")
        
        # Initialize embeddings
        logger.info("Loading embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'}
        )
        
        # Load documents
        if os.path.exists(docs_path):
            logger.info(f"Loading documents from {docs_path}...")
            loader = DirectoryLoader(
                docs_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents")
        else:
            logger.info(f"Path {docs_path} not found. Using sample documents...")
            documents = create_sample_documents()
        
        if not documents:
            logger.warning("No documents found. Creating sample documents...")
            documents = create_sample_documents()
        
        # Split documents into chunks
        logger.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=500
        )
        docs = text_splitter.split_documents(documents)
        logger.info(f"Created {len(docs)} document chunks")
        
        # Create FAISS index
        logger.info("Creating FAISS index...")
        vector_store = FAISS.from_documents(docs, embeddings)
        
        # Save index
        logger.info(f"Saving index to {output_path}...")
        os.makedirs(output_path, exist_ok=True)
        vector_store.save_local(output_path)
        logger.info(f"✅ FAISS index created and saved to {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating FAISS index: {str(e)}")
        raise

def create_sample_documents():
    """Create sample documents for testing"""
    sample_docs = [
        Document(page_content="Artificial Intelligence (AI) is the simulation of human intelligence by machines. It includes machine learning, deep learning, and neural networks."),
        Document(page_content="Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed."),
        Document(page_content="Deep Learning uses neural networks with multiple layers to analyze various factors of data. It's particularly effective for image and speech recognition."),
        Document(page_content="Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language."),
        Document(page_content="Computer Vision enables computers to derive meaningful information from digital images, videos and other visual inputs."),
        Document(page_content="The Transformer architecture revolutionized NLP and led to models like BERT, GPT, and T5."),
        Document(page_content="Python is the most popular programming language for AI and machine learning due to its simplicity and extensive libraries."),
        Document(page_content="RAG (Retrieval-Augmented Generation) combines information retrieval with text generation for more accurate AI responses."),
        Document(page_content="FAISS is a library for efficient similarity search and clustering of dense vectors, developed by Facebook AI Research."),
        Document(page_content="Large Language Models (LLMs) like GPT and Llama are trained on vast amounts of text data to understand and generate human-like text."),
    ]
    return sample_docs

if __name__ == "__main__":
    if not HF_API_KEY:
        logger.error("❌ HUGGINGFACE_API_KEY is required! Set it in .env file.")
        exit(1)
    
    # Create FAISS index
    index_path = create_faiss_index_from_documents(DOCS_PATH, INDEX_OUTPUT_PATH)
    logger.info(f"Index saved at: {index_path}")
