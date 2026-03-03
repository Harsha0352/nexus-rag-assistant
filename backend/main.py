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
base_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path=env_path)
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
                pinecone_retriever = pinecone_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
                pinecone_rag_chain = build_rag_chain_custom(pinecone_retriever)
                logger.info("✓ Pinecone initialized and RAG chain built.")
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
                # FACTUAL RESCUE: Switch to similarity search for better factual recall in clinical STGs
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
                rag_chain = build_rag_chain()
                logger.info("✓ Previous FAISS index recovered (Similarity Search active).")
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
    # Include clear page numbering for medical reference accuracy
    return "\n\n".join(f"[Source: PDF Page {doc.metadata.get('page', 'unknown')}]\n{doc.page_content.strip()}" for doc in docs)

def build_rag_chain():
    global retriever, llm
    if retriever is None or llm is None:
        return None
    
    prompt = ChatPromptTemplate.from_template(
        """You are an advanced medical assistant. Your task is to provide precise answers based ONLY on the provided context.

Context:
{context}

Question:
{question}

Instructions:
1. If the answer is in the context, provide it clearly and mention the source page found in the context.
2. If the answer is not in the context, state: "The provided medical guidelines do not contain information on this specific topic."
3. Do not use outside knowledge. Rely exclusively on the provided medical text.

Answer based on specific medical guidelines:"""
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
        """You are an advanced medical assistant. Your task is to provide precise answers based ONLY on the provided context.

Context:
{context}

Question:
{question}

Instructions:
1. If the answer is in the context, provide it clearly and mention the source page found in the context.
2. If the answer is not in the context, state: "The provided medical guidelines do not contain information on this specific topic."
3. Do not use outside knowledge. Rely exclusively on the provided medical text.

Answer based on specific medical guidelines:"""
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
    """FACTUAL RESCUE: Robust ingestion with similarity search and memory safety."""
    global vectorstore, retriever, rag_chain, embeddings, is_processing, processing_stats
    global pinecone_retriever, pinecone_rag_chain
    
    error_occurred = False
    
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
        # FACTUAL RESCUE: Smaller, cleaner chunks for clinical accuracy
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600, 
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        for i, temp_path in enumerate(temp_paths):
            try:
                if not os.path.exists(temp_path):
                    continue
                
                logger.info(f"Processing: {file_names[i]}")
                processing_stats["current_file"] = file_names[i]
                
                # Improved page accounting
                if processing_stats["total_pages"] == 0:
                    processing_stats["total_pages"] = 300 # Initial estimate
                
                loader = PyPDFLoader(temp_path)
                batch_docs = []
                p_count = 0
                
                # Robust Page iteration
                try:
                    pages = loader.lazy_load()
                except Exception as load_err:
                    logger.error(f"Failed to initialize loader for {file_names[i]}: {load_err}")
                    error_occurred = True
                    continue

                for page in pages:
                    try:
                        batch_docs.append(page)
                        p_count += 1
                        processing_stats["processed_pages"] += 1
                        
                        # Update total pages dynamically if we exceed estimate
                        if processing_stats["processed_pages"] > processing_stats["total_pages"]:
                            processing_stats["total_pages"] = processing_stats["processed_pages"] + 50 # Add buffer
                        
                        # 15-page micro-batches for a balance of speed and memory safety
                        if len(batch_docs) >= 15:
                            _process_batch(batch_docs, p_count, embeddings, splitter)
                            batch_docs = []
                            time.sleep(0.5) # Slight pause for API stability
                    except Exception as page_err:
                        logger.error(f"Error on page {p_count} in {file_names[i]}: {page_err}")
                        error_occurred = True
                        continue

                # Final Batch for this file
                if batch_docs:
                    _process_batch(batch_docs, p_count, embeddings, splitter)
                    batch_docs = []

            except Exception as e:
                logger.error(f"Error reading PDF {temp_path}: {e}")
                error_occurred = True
            finally:
                if os.path.exists(temp_path):
                    try: os.remove(temp_path)
                    except: pass

        # Final Reconciliation
        if vectorstore:
            vectorstore.save_local("faiss_index")
            logger.info("✓ Final neural map saved to disk.")
            
        processing_stats["total_pages"] = processing_stats["processed_pages"]
        if error_occurred:
            logger.warning("Ingestion finished with some non-fatal errors.")
            
    except Exception as e:
        logger.error(f"Global Ingestion Error: {e}")
    finally:
        is_processing = False
        processing_stats["current_file"] = "Completed"

def _process_batch(batch_docs, p_count, embeddings, splitter):
    """Helper to process a batch of documents."""
    global vectorstore, retriever, rag_chain
    global pinecone_vectorstore, pinecone_retriever, pinecone_rag_chain

    try:
        logger.info(f"Digitizing Page Batch ending at {p_count}...")
        chunks = splitter.split_documents(batch_docs)
        
        if vectorstore is None:
            vectorstore = FAISS.from_documents(chunks, embeddings)
        else:
            vectorstore.add_documents(chunks)
        
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
        rag_chain = build_rag_chain()
        
        if pinecone_vectorstore:
            try:
                pinecone_vectorstore.add_documents(chunks)
                pinecone_retriever = pinecone_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
                pinecone_rag_chain = build_rag_chain_custom(pinecone_retriever)
            except Exception as pc_err:
                logger.error(f"Pinecone Sync Error: {pc_err}")
    except Exception as batch_err:
        logger.error(f"Batch Processing Error: {batch_err}")
        # We don't re-raise here to keep the background thread alive

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
