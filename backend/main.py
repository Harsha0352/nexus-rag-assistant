import os
import threading
import logging
import shutil
import time
import tempfile
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try robust imports
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import (
        HuggingFaceEndpointEmbeddings,
        HuggingFaceEndpoint,
        ChatHuggingFace,
    )
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from operator import itemgetter
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore
except ImportError as e:
    logger.error(f"Import Error: {e}")
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

# ---------------- ENV ----------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ---------------- GLOBALS ----------------
llm = None
embeddings = None
vectorstore = None
retriever = None
rag_chain = None
pc = None
pinecone_vectorstore = None
pinecone_retriever = None
pinecone_rag_chain = None
is_processing = False  # Track background PDF synchronization
processing_stats = {"total_pages": 0, "processed_pages": 0, "current_file": ""}

# ---------------- STARTUP ----------------
def load_models():
    global llm, embeddings, vectorstore, retriever, rag_chain

    logger.info("Initializing AI models in background...")
    
    try:
        # Use a standardized embedding model
        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=HF_TOKEN,
        )

        llm_endpoint = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            task="text-generation",
            huggingfacehub_api_token=HF_TOKEN,
            temperature=0.3, # Lowered from 0.5 for better factuality
            max_new_tokens=1024, # Increased for more detailed answers
        )

        llm = ChatHuggingFace(llm=llm_endpoint)
        logger.info("✓ AI Models (API-based) initialized.")

        # Initialize Pinecone
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
        if pinecone_api_key and pinecone_index_name:
            try:
                global pc, pinecone_vectorstore, pinecone_retriever, pinecone_rag_chain
                pc = Pinecone(api_key=pinecone_api_key)
                
                # Check if index exists, else create it
                existing_indexes = [index.name for index in pc.list_indexes()]
                if pinecone_index_name not in existing_indexes:
                    logger.info(f"Creating Pinecone index: {pinecone_index_name}")
                    pc.create_index(
                        name=pinecone_index_name,
                        dimension=384, # all-MiniLM-L6-v2 dimension
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                
                # Wait for index to be ready
                logger.info(f"Waiting for Pinecone index '{pinecone_index_name}' to be ready...")
                import time
                wait_time = 0
                while not pc.describe_index(pinecone_index_name).status['ready'] and wait_time < 60:
                    time.sleep(2)
                    wait_time += 2
                
                pinecone_vectorstore = PineconeVectorStore(
                    index_name=pinecone_index_name,
                    embedding=embeddings,
                    pinecone_api_key=pinecone_api_key
                )
                logger.info("✓ Pinecone initialized.")
            except Exception as e:
                logger.error(f"Pinecone initialization failed: {e}")

        # Optional: Load previous index if exists
        if os.path.exists("faiss_index"):
            try:
                vectorstore = FAISS.load_local(
                    "faiss_index",
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                # Increased k from 6 to 10 for better coverage in large PDFs
                retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 30})
                rag_chain = build_rag_chain()
                logger.info("✓ Previous FAISS index recovered.")
            except Exception as e:
                logger.warning(f"Could not load old index: {e}")
    except Exception as e:
        logger.error(f"Background initialization failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start loading models in a background thread
    thread = threading.Thread(target=load_models, daemon=True)
    thread.start()
    yield

app = FastAPI(lifespan=lifespan)

# -------- CORS -------- #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- SERVE FRONTEND -------- #
@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    try:
        # Use path relative to main.py for reliability on all platforms
        base_path = os.path.dirname(os.path.abspath(__file__))
        html_path = os.path.join(base_path, "frontend", "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Frontend error: {e}")
        return f"<h1>Nexus AI Online</h1><p>Frontend file not found at {html_path}</p>"

# ---------------- HELPERS ----------------
def format_docs(docs):
    if not docs:
        return "No relevant context found."
    return "\n\n".join(f"[Source: Page {doc.metadata.get('page', 'unknown')}]\n{doc.page_content}" for doc in docs)

def build_rag_chain():
    global retriever, llm
    if retriever is None or llm is None:
        return None
    
    prompt = ChatPromptTemplate.from_template(
        """You are an advanced medical and professional assistant. Your primary goal is to provide accurate, detailed, and context-aware information.

Analyze the provided context thoroughly. If the question asks for something not directly present but can be inferred reliably, do so cautiously. If the answer is truly not in the context, state that clearly.

Note: Digitized PDF text may have formatting issues. Interpret technical terms and concepts flexibly.

Context:
{context}

Question:
{question}

Complete and detailed answer based ONLY on context:"""
    )

    return (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

def build_rag_chain_custom(custom_retriever):
    global llm
    if custom_retriever is None or llm is None:
        return None
    
    prompt = ChatPromptTemplate.from_template(
        """You are an advanced medical and professional assistant. Your primary goal is to provide accurate, detailed, and context-aware information.

Analyze the provided context thoroughly. If the question asks for something not directly present but can be inferred reliably, do so cautiously. If the answer is truly not in the context, state that clearly.

Note: Digitized PDF text may have formatting issues. Interpret technical terms and concepts flexibly.

Context:
{context}

Question:
{question}

Complete and detailed answer based ONLY on context:"""
    )

    return (
        {
            "context": itemgetter("question") | custom_retriever | format_docs,
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

def clear_existing_data():
    """Clears old data to ensure new uploads are not mixed with old results."""
    global vectorstore, retriever, rag_chain, processing_stats
    vectorstore = None
    retriever = None
    rag_chain = None
    processing_stats = {"total_pages": 0, "processed_pages": 0, "current_file": ""}
    
    global pinecone_vectorstore, pinecone_retriever, pinecone_rag_chain
    if pinecone_vectorstore:
        try:
            # We don't delete the index, just the content
            pinecone_vectorstore.delete(delete_all=True)
            logger.info("✓ Pinecone index cleared.")
        except Exception as e:
            logger.warning(f"Pinecone cleanup error: {e}")
            
    pinecone_retriever = None
    pinecone_rag_chain = None

    if os.path.exists("faiss_index"):
        try:
            shutil.rmtree("faiss_index")
            logger.info("✓ Old FAISS index cleared.")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

def process_pdf_background(temp_paths: List[str], file_names: List[str]):
    """Final robustness fix: 10-page micro-batching and 700-char chunks for better vector accuracy."""
    global vectorstore, retriever, rag_chain, embeddings, is_processing, processing_stats
    global pinecone_retriever, pinecone_rag_chain
    
    is_processing = True
    logger.info("Starting Optimized Ingestion Task...")
    
    # 1. Wait for embeddings (Extended to 120s for slow cold starts)
    wait_count = 0
    while embeddings is None and wait_count < 24:
        logger.info(f"AI Brain warming up... ({wait_count * 5}s)")
        time.sleep(5)
        wait_count += 1
        
    if embeddings is None:
        logger.error("Sync Critical Failure: AI Brain did not initialize.")
        is_processing = False
        return

    try:
        # Optimized chunk size for all-MiniLM-L6-v2 (max 256 tokens)
        # ~700-800 chars is usually safe for ~200 tokens
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700, 
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        for i, temp_path in enumerate(temp_paths):
            try:
                if not os.path.exists(temp_path):
                    continue
                
                processing_stats["current_file"] = file_names[i]
                loader = PyPDFLoader(temp_path)
                
                # Try to count total pages if possible
                try:
                    all_pages = list(loader.lazy_load())
                    processing_stats["total_pages"] += len(all_pages)
                except:
                    pass
                
                batch_docs = []
                p_count = 0
                
                # Reload for lazy loading
                loader = PyPDFLoader(temp_path)
                for page in loader.lazy_load():
                    batch_docs.append(page)
                    p_count += 1
                    processing_stats["processed_pages"] += 1
                    
                    # 10-page micro-batches for absolute memory safety
                    if len(batch_docs) >= 10:
                        logger.info(f"Digitizing pages up to {p_count} in {file_names[i]}...")
                        chunks = splitter.split_documents(batch_docs)
                        if vectorstore is None:
                            vectorstore = FAISS.from_documents(chunks, embeddings)
                        else:
                            vectorstore.add_documents(chunks)
                        
                        # Increased k for better recall
                        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 30})
                        rag_chain = build_rag_chain()
                        
                        # Dual Indexing: Pinecone
                        if pinecone_vectorstore:
                            pinecone_vectorstore.add_documents(chunks)
                            pinecone_retriever = pinecone_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 30})
                            pinecone_rag_chain = build_rag_chain_custom(pinecone_retriever)

                        batch_docs = [] # GC RAM immediately
                        time.sleep(0.3) # Prevent API rate limit triggers

                # Final Batch
                if batch_docs:
                    chunks = splitter.split_documents(batch_docs)
                    if vectorstore is None:
                        vectorstore = FAISS.from_documents(chunks, embeddings)
                    else:
                        vectorstore.add_documents(chunks)
                    
                    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 30})
                    rag_chain = build_rag_chain()
                    
                    if pinecone_vectorstore:
                        pinecone_vectorstore.add_documents(chunks)
                        pinecone_retriever = pinecone_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 30})
                        pinecone_rag_chain = build_rag_chain_custom(pinecone_retriever)

            except Exception as e:
                logger.error(f"Error reading PDF {temp_path}: {e}")
            finally:
                if os.path.exists(temp_path):
                    try: os.remove(temp_path)
                    except: pass

        if vectorstore:
            vectorstore.save_local("faiss_index")
            logger.info("✓ Universal Ingestion Complete.")
            
    except Exception as e:
        logger.error(f"Background Loop Error: {e}")
    finally:
        is_processing = False
        processing_stats["current_file"] = "Completed"

# ---------------- ENDPOINTS ----------------
@app.post("/upload-pdf/")
async def upload_pdf(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    global is_processing

    if not files:
        raise HTTPException(status_code=400, detail="Document missing.")

    if is_processing:
        raise HTTPException(status_code=429, detail="AI is currently busy with another document. Please wait.")

    # Reset old data immediately
    clear_existing_data()

    temp_paths = []
    file_names = []
    for file in files:
        # Use tempfile for guaranteed unique/writeable paths
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_paths.append(tmp.name)
            file_names.append(file.filename)

    # Trigger final robust background sync
    background_tasks.add_task(process_pdf_background, temp_paths, file_names)

    return {"message": "Neural Link synchronization initiated. Digestion progress can be monitored at /status endpoint. ✅"}

@app.get("/status")
async def get_status():
    global is_processing, processing_stats
    progress = 0
    if processing_stats["total_pages"] > 0:
        progress = round((processing_stats["processed_pages"] / processing_stats["total_pages"]) * 100, 2)
    
    return {
        "is_processing": is_processing,
        "stats": processing_stats,
        "progress_percentage": progress
    }

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    global rag_chain, is_processing
    
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")
        
    # Check if we have NOTHING yet
    if rag_chain is None:
        if is_processing:
            raise HTTPException(status_code=503, detail="AI Brain is still establishing the Neural Link. Please wait a few more seconds.")
        raise HTTPException(status_code=404, detail="No active document map found. Please upload a PDF to begin.")
    
    try:
        # Dynamic warning
        prefix = ""
        if is_processing:
            prefix = "[AI Sync in Progress: Only partially digitized results available]\n\n"

        # Compare results
        faiss_response = "FAISS not initialized"
        if rag_chain:
            faiss_response = rag_chain.invoke({"question": question})
        
        pinecone_response = "Pinecone not initialized"
        if pinecone_rag_chain:
            pinecone_response = pinecone_rag_chain.invoke({"question": question})

        final_answer = f"{prefix}**[FAISS MODEL ANSWER]**\n{faiss_response}\n\n---\n\n**[PINECONE MODEL ANSWER]**\n{pinecone_response}"
        
        return {"answer": final_answer}
    except Exception as e:
        logger.error(f"Ask Error: {e}")
        raise HTTPException(status_code=500, detail="Neural link interrupted. Please try again.")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
