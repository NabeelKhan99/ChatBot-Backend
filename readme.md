# Smart FAQ 🤖  

An AI-powered chatbot that acts as a **Smart FAQ assistant**, answering user queries directly within your site. Built with a **React + Vite + TypeScript frontend** and a **FastAPI backend powered by Hugging Face’s google/flan-t5-small**. Serving it on my portfolio site using **Railway cloud services**.

---

## 🚀 Features (Backend - Implemented)
- **FastAPI Endpoint `/chat`** – Handles user queries and returns AI-generated responses.  
- **Google/Flan-T5-Small Integration** – Provides context-aware answers using Hugging Face model.  
- **Fallback Humor** – Returns a random joke from `jokes.txt` when no confident answer is found.  
- **CORS Enabled** – Ensures smooth frontend-backend communication.  

---

## 🛠 Upcoming Features
- **Knowledge Base Extraction** – Automatic text extraction from site `.tsx` pages into structured `.txt` files.  
- **Grounded Responses** – Chatbot will use site content as its knowledge base for more accurate answers.  
- **Expanded Model Options** – Potential upgrades to larger language models for improved accuracy.  

---

## 📦 Tech Stack
- **Frontend:** React, Vite, TypeScript  
- **Backend:** FastAPI, Hugging Face Transformers (Flan-T5)  
- **Other:** Node.js scripts (for upcoming knowledge extraction), jokes.json  

---

## ⚙️ Setup

### Backend
```bash
# Install dependencies
pip install -r requirements.txt  

# Run FastAPI server
uvicorn main:app --reload
