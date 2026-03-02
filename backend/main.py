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
is_processing = False  # Track background PDF synchronization

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
            temperature=0.5,
            max_new_tokens=512,
        )

        llm = ChatHuggingFace(llm=llm_endpoint)
        logger.info("✓ AI Models (API-based) initialized.")

        # Optional: Load previous index if exists
        if os.path.exists("faiss_index"):
            try:
                vectorstore = FAISS.load_local(
                    "faiss_index",
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
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
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "<h1>Nexus AI Online</h1><p>Frontend file not found.</p>"

# ---------------- HELPERS ----------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain():
    global retriever, llm
    if retriever is None or llm is None:
        return None
    
    prompt = ChatPromptTemplate.from_template(
        """You are a medical and professional assistant. Answer the question based ONLY on the provided context. 

Note: The context may contain slight formatting errors (like missing spaces between words) due to PDF digitization. Be flexible and search for the concepts described even if names are slightly merged.

Context:
{context}

Question:
{question}

Answer:"""
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

def clear_existing_data():
    """Clears old data to ensure new uploads are not mixed with old results."""
    global vectorstore, retriever, rag_chain
    vectorstore = None
    retriever = None
    rag_chain = None
    if os.path.exists("faiss_index"):
        try:
            shutil.rmtree("faiss_index")
            logger.info("✓ Old FAISS index cleared.")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

def process_pdf_background(temp_paths: List[str]):
    """Final robustness fix: 10-page micro-batching and 1000-char chunks."""
    global vectorstore, retriever, rag_chain, embeddings, is_processing
    
    is_processing = True
    logger.info("Starting Ultra-Stable Ingestion Task...")
    
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
        # Hyper-conservative settings for huge 500-page files on 512MB RAM
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        
        for temp_path in temp_paths:
            try:
                if not os.path.exists(temp_path):
                    continue
                
                loader = PyPDFLoader(temp_path)
                batch_docs = []
                p_count = 0
                
                for page in loader.lazy_load():
                    batch_docs.append(page)
                    p_count += 1
                    
                    # 10-page micro-batches for absolute memory safety
                    if len(batch_docs) >= 10:
                        logger.info(f"Digitizing pages up to {p_count}...")
                        chunks = splitter.split_documents(batch_docs)
                        if vectorstore is None:
                            vectorstore = FAISS.from_documents(chunks, embeddings)
                        else:
                            vectorstore.add_documents(chunks)
                        
                        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6})
                        rag_chain = build_rag_chain()
                        batch_docs = [] # GC RAM immediately
                        time.sleep(0.5) # Prevent API rate limit triggers

                # Final Batch
                if batch_docs:
                    chunks = splitter.split_documents(batch_docs)
                    if vectorstore is None:
                        vectorstore = FAISS.from_documents(chunks, embeddings)
                    else:
                        vectorstore.add_documents(chunks)
                    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6})
                    rag_chain = build_rag_chain()

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
    for file in files:
        # Use tempfile for guaranteed unique/writeable paths
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_paths.append(tmp.name)

    # Trigger final robust background sync
    background_tasks.add_task(process_pdf_background, temp_paths)

    return {"message": "Neural Link synchronization initiated. Digestion will take 1-3 minutes for large documents ✅"}

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

        response = rag_chain.invoke({"question": question})
        return {"answer": f"{prefix}{response}"}
    except Exception as e:
        logger.error(f"Ask Error: {e}")
        raise HTTPException(status_code=500, detail="Neural link interrupted. Please try again.")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
