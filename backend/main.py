import os
import threading
import logging
import shutil
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

        if os.path.exists("faiss_index"):
            vectorstore = FAISS.load_local(
                "faiss_index",
                embeddings,
                allow_dangerous_deserialization=True
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            rag_chain = build_rag_chain()
            logger.info("✓ FAISS index loaded.")
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
        "Answer the question based only on the context below.\n\nContext:\n{context}\n\nQuestion:\n{question}"
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
            logger.info("✓ Old FAISS index cleared from disk.")
        except Exception as e:
            logger.warning(f"Could not clear old index: {e}")

async def process_pdf_background(temp_paths: List[str]):
    """Handles heavy PDF processing and vectorization in batches for memory safety."""
    global vectorstore, retriever, rag_chain, embeddings, is_processing
    
    is_processing = True
    logger.info(f"Background synchronization started for {len(temp_paths)} files.")
    
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
        
        for temp_path in temp_paths:
            try:
                loader = PyPDFLoader(temp_path)
                # Load document pages
                pages = loader.load()
                
                # Process in batches of 50 pages to stay under 512MB RAM
                batch_size = 50
                for i in range(0, len(pages), batch_size):
                    batch_pages = pages[i : i + batch_size]
                    logger.info(f"Processing batch {i//batch_size + 1} of {temp_path} ({len(batch_pages)} pages)...")
                    
                    chunks = splitter.split_documents(batch_pages)
                    
                    if chunks and embeddings:
                        if vectorstore is None:
                            vectorstore = FAISS.from_documents(chunks, embeddings)
                        else:
                            vectorstore.add_documents(chunks)
                        
                        # Periodically update state to show progress
                        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                        rag_chain = build_rag_chain()
                        
            except Exception as e:
                logger.error(f"Error processing {temp_path}: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        if vectorstore:
            # Save the final index to disk for persistence
            vectorstore.save_local("faiss_index")
            logger.info("✓ PDF Synchronization complete and saved to disk.")
        else:
            logger.warning("No data was vectorized.")
            
    except Exception as e:
        logger.error(f"Critical error in background PDF task: {e}")
    finally:
        is_processing = False

# ---------------- ENDPOINTS ----------------
@app.post("/upload-pdf/")
async def upload_pdf(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    global is_processing

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    if is_processing:
        raise HTTPException(status_code=429, detail="AI is currently digesting another document. Please wait.")

    # 1. Clear ANY old state immediately (Resume vs Health PDF)
    clear_existing_data()

    temp_paths = []
    for file in files:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        temp_paths.append(temp_path)

    # 2. Start background batch processing
    background_tasks.add_task(process_pdf_background, temp_paths)

    return {"message": "Large document ingestion started. AI is digesting content in batches to ensure maximum accuracy ✅"}

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    global rag_chain, is_processing
    
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")
        
    if rag_chain is None:
        if is_processing:
            raise HTTPException(status_code=503, detail="AI is still establishing its neural link with the new document. Please wait about 30-60 seconds.")
        raise HTTPException(status_code=503, detail="No document maps found. Please upload a PDF.")
    
    try:
        # If still processing but we have some data, warn the user
        processing_warning = ""
        if is_processing:
            processing_warning = "[Note: AI is still digesting the full document, answers might be incomplete until sync finishes]\n\n"

        response = rag_chain.invoke({"question": question})
        return {"answer": f"{processing_warning}{response}"}
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        raise HTTPException(status_code=500, detail="Neural link interrupted. Please try re-phrasing.")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
