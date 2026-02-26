# Voice RAG System - Complete Voice Conversation Pipeline

## 📋 Overview

Your system now has a complete end-to-end voice conversation pipeline:

```
User speaks → Microphone → Audio captured → Backend:
  1. Speech-to-Text (Whisper API)
  2. RAG Model Processing (Llama 3.1 + FAISS)
  3. Text-to-Speech (HF TTS API)
→ Audio response → Browser plays audio
```

## 🚀 Getting Started

### 1. **Create the FAISS Index** (First time only)

```powershell
. C:\Users\Vamsi\Desktop\task21\venv\Scripts\Activate.ps1
cd C:\Users\Vamsi\Desktop\task21\backend
python create_faiss_index.py
```

This creates a vector index for faster document retrieval.

### 2. **Start the Backend Server**

```powershell
. C:\Users\Vamsi\Desktop\task21\venv\Scripts\Activate.ps1
cd C:\Users\Vamsi\Desktop\task21\backend
python main.py
```

Wait for the message: `Application startup complete`

### 3. **Open the Frontend**

Open your browser and go to:
```
http://localhost:8000
```

You should see the Voice RAG Assistant interface.

## 🎤 How to Use

1. **Click "Start Recording"** - Allows browser to access your microphone
2. **Speak your question** - "Why is Python used for machine learning?"
3. **Click "Stop Recording"** - Sends audio to backend
4. **Wait for processing** - Backend processes through the pipeline:
   - Converts speech to text
   - Retrieves relevant documents
   - Generates AI response using Llama 3.1
   - Converts response to speech
5. **Listen to the response** - Audio plays automatically

## 📁 Project Structure

```
task21/
├── backend/
│   ├── main.py                    # FastAPI server
│   ├── create_faiss_index.py      # Index creation script
│   ├── .env                       # API keys
│   ├── requirements.txt           # Dependencies
│   ├── faiss_index/              # Generated index (after running create_faiss_index.py)
│   │   ├── index.faiss
│   │   └── index.pkl
│   └── books/                     # PDF documents (optional)
│
└── frontend/
    └── index.html                # Voice UI interface
```

## 🔧 Key Endpoints

### Voice Processing
```bash
POST /process-voice
- Upload audio file
- Returns: JSON with transcribed text, response text, and audio file
- Headers: X-Transcribed-Text, X-Response-Text
```

### Text Query (For testing)
```bash
POST /query
Body: {"text": "your question"}
Returns: {"query": "...", "response": "..."}
```

### Health Check
```bash
GET /health
Returns: System status and configuration info
```

### Documentation
```bash
GET /docs          # Swagger UI
GET /redoc         # ReDoc
```

## 🛠️ Technologies Used

- **Backend**: FastAPI, Python
- **LLM**: Llama 3.1 70B via Hugging Face
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector DB**: FAISS
- **Speech-to-Text**: Whisper (OpenAI via HF API)
- **Text-to-Speech**: Facebook MMS TTS via HF API
- **Frontend**: HTML5, Web Audio API, Fetch API

## ✅ Features

- ✅ Real-time microphone recording
- ✅ Speech-to-Text conversion
- ✅ RAG-based response generation
- ✅ Text-to-Speech response audio
- ✅ FAISS vector index for fast retrieval
- ✅ Persistent chat history
- ✅ Document management endpoints
- ✅ Beautiful, responsive UI
- ✅ Real-time status indicators

## 🔐 Environment Setup

Make sure your `.env` file contains:
```
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxxxx
```

Get your free API key at: https://huggingface.co/settings/tokens

## 📝 Example Usage

### Speaking in the UI:
```
User: "What is machine learning?"

Flow:
1. Audio → Whisper → "What is machine learning?"
2. Text → FAISS → Retrieves relevant documents
3. RAG → Llama 3.1 → "Machine Learning is a subset of AI..."
4. Text → TTS → Audio file
5. Browser → Plays audio response
```

## 🎯 Troubleshooting

### "Cannot connect to backend"
- Make sure backend is running: `python main.py`
- Check that API key is set in `.env`
- Verify port 8000 is not already in use

### "Microphone permission denied"
- Check browser microphone permissions
- Refresh the page and allow access

### "FAISS index not found"
- Run: `python create_faiss_index.py`
- System will use sample documents if index missing

### "API Error - Invalid token"
- Verify HUGGINGFACE_API_KEY in `.env`
- Make sure token is active in HF settings

## 📚 Adding Your Own Documents

### Method 1: PDF Files
1. Add PDFs to `backend/books/` folder
2. Run: `python create_faiss_index.py`
3. Restart the server

### Method 2: Text via API
```bash
POST /add-document
Body: {"text": "Your custom content here"}
```

## 🔊 Audio Formats

- **Input**: WAV, MP3, OGG, FLAC (browser supported formats)
- **Output**: FLAC audio (efficient for speech)

## 💡 Tips

1. Speak clearly for better transcription
2. Ask specific questions for better results
3. FAISS index speeds up retrieval significantly
4. Check `/docs` for interactive API testing
5. Monitor backend logs for debugging

## 📞 Support

If you encounter issues:
1. Check backend logs in terminal
2. Verify all environment variables
3. Ensure dependencies are installed: `pip install -r requirements.txt`
4. Clear browser cache and refresh

---

**Happy voice chatting! 🎉**
