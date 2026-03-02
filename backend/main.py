import os
import threading
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
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
    # Fallback to older paths if needed
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

# ---------------- ENV ----------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

# ---------------- APP ----------------
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

# ---------------- GLOBALS ----------------
llm = None
embeddings = None
vectorstore = None
retriever = None
rag_chain = None

# ---------------- STARTUP ----------------
def load_models():
    global llm, embeddings, vectorstore, retriever, rag_chain

    logger.info("Initializing AI models in background...")
    
    try:
        # Revert to API base embeddings for Render Free Tier (512MB RAM)
        # torch/sentence-transformers are too heavy for local execution on free tier
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

# ---------------- ENDPOINTS ----------------
@app.post("/upload-pdf/")
async def upload_pdf(files: List[UploadFile] = File(...)):
    global vectorstore, retriever, rag_chain, embeddings

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    all_chunks = []
    for file in files:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        try:
            loader = PyPDFLoader(temp_path)
            # Increased chunk_size to 2000 to reduce the number of API calls for large PDFs
            # This balances upload speed and memory usage on Render Free Tier
            chunks = loader.load_and_split(RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300))
            all_chunks.extend(chunks)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    if all_chunks:
        if embeddings is None:
            raise HTTPException(status_code=503, detail="AI models are still initializing.")
        vectorstore = FAISS.from_documents(all_chunks, embeddings)
        retriever = vectorstore.as_retriever()
        rag_chain = build_rag_chain()

    return {"message": "PDF processed successfully ✅"}

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    global rag_chain
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="AI link is not initialized yet.")
    
    try:
        response = rag_chain.invoke({"question": question})
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Respect PORT environment variable for Render
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
