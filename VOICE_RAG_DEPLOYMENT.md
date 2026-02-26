# Voice RAG System - Complete Deployment Guide

## ✅ Current Setup

### Backend (FastAPI)
- **Text Chat**: Meta-Llama-3.1-70B-Instruct for RAG responses
- **STT (Speech-to-Text)**: OpenAI Whisper Large v2 via HuggingFace API
- **Text Generation**: Meta-Llama-3.1-70B-Instruct  (already set)
- **TTS (Text-to-Speech)**: ESPnet Glow-TTS (for future audio responses)

### Frontend  
- **Audio Input**: Web Audio API - Records user voice as WebM
- **Audio Processing**: Sends to `/ask-voice/` endpoint for transcription
- **Audio Output**: Browser Web Speech API (native speech synthesis)

## 📝 Endpoints

### 1. `/ask-voice/` - Voice Question Endpoint
- **Method**: POST
- **Input**: Audio file (WAV, MP3, OGG, M4A, WebM)
- **Process**:
  1. Transcribes audio → text (Whisper API)
  2. Gets answer from RAG chain (Llama 3.1)
  3. Returns JSON with transcription & answer
- **Output**: `{"answer": "...", "transcription": "..."}`
- **Audio**: Browser speaks the answer using Web Speech API

### 2. `/ask/` - Text Question Endpoint  
- **Method**: POST
- **Input**: `question` form parameter
- **Output**: `{"answer": "..."}`

### 3. `/upload-pdf-multi/` - Upload Multiple PDFs
- **Method**: POST
- **Input**: Multiple PDF files
- **Process**: Creates FAISS embeddings index
- **Output**: Success message

### 4. `/` - Frontend HTML
- **Method**: GET
- **Output**: Main UI with mic recording feature

## 🚀 How to Use

###  1. Start Backend
```powershell
cd C:\Users\Vamsi\Desktop\task21\backend
uvicorn main:app --reload
```

### 2. Open Frontend
```
http://localhost:8000
```

### 3. Upload PDFs
- Drag & drop or browse PDFs
- Click "Submit & Process"
- Wait for embedding index to be created

### 4. Ask Questions

**Text Mode**: Type in text box → Click "Send"

**Voice Mode**: Click 🎤 button → Speak question → Wait for transcription → Listen to response

## 🔧 Troubleshooting

### "Internal Server Error" on Voice Input
1. Check backend logs for exact error
2. Verify HuggingFace API key is valid
3. Check if audio file is in correct format (WebM, WAV, MP3, OGG, M4A)

### No Transcription Result
- Speak clearly and slowly
- Check if Whisper API quota is available
- Verify HF_TOKEN is set in .env

### Browser Speech Not Working
- Allow browser permission for microphone & speakers
- Check if speechSynthesis is available in browser (Chrome, Edge, Safari)
- Some browsers limit speech synthesis to HTTPS (try localhost)

## 🔐 Environment Variables (.env)

```
HUGGINGFACE_API_KEY=hf_xxxxxxxxxxxxx
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxx  # Alternative name
```

Get free API key: https://huggingface.co/settings/tokens

## 📊 Flow Diagram

```
User speaks
    ↓
Browser records WebM audio
    ↓ 
POST to /ask-voice/ with audio file
    ↓
Backend: Transcribe with Whisper API
    ↓
Backend: Q&A with RAG (Llama 3.1 + FAISS)
    ↓
Backend: Returns JSON {"answer": "...", "transcription": "..."}
    ↓
Frontend: Displays answer text
    ↓
Frontend: Speaks answer using Web Speech API
    ↓
User hears response ✓
```

## 🎯 Future Enhancements

- [ ] Store conversation history
- [ ] Add language selection  
- [ ] Implement proper TTS audio response instead of browser synthesis
- [ ] Add real-time transcription progress
- [ ] Support for audio file uploads

---

**Status**: ✅ Fully functional voice RAG system with Whisper STT + Llama RAG + Web Speech TTS
