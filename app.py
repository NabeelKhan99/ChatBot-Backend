import os
import random
import re
import time
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from functools import lru_cache

app = FastAPI()

# Allow only your Netlify frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://nabeel-saeed.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    page: str = "/"  # optional page info

# ---------------------------
# Lazy loading (cache models)
# ---------------------------
@lru_cache(maxsize=1)
def get_qa_pipeline():
    model_name = "google/flan-t5-small"  # use flan-t5-nano if still too heavy
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

@lru_cache(maxsize=1)
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------
# Knowledge base loading
# ---------------------------
knowledge_chunks = []
knowledge_files = []
if os.path.exists("knowledge_base"):
    for fname in os.listdir("knowledge_base"):
        if fname.endswith(".txt"):
            text = open(os.path.join("knowledge_base", fname), encoding="utf-8").read()
            text = re.sub(r'(\d+)\s', r'\n\1- ', text)  # add dash after numbers
            knowledge_chunks.append(text.strip())
            knowledge_files.append(fname)

knowledge_embeddings = (
    get_embedder().encode(knowledge_chunks, convert_to_tensor=True) if knowledge_chunks else None
)

# ---------------------------
# Jokes fallback
# ---------------------------
jokes = []
if os.path.exists("jokes.txt"):
    with open("jokes.txt", encoding="utf-8") as f:
        jokes = [line.strip() for line in f.readlines() if line.strip()]

# ---------------------------
# Simple rate limiter
# ---------------------------
requests_per_ip = {}
RATE_LIMIT = 5   # requests
WINDOW = 60      # seconds

def check_rate_limit(ip: str):
    now = time.time()
    if ip not in requests_per_ip:
        requests_per_ip[ip] = []
    requests_per_ip[ip] = [t for t in requests_per_ip[ip] if now - t < WINDOW]

    if len(requests_per_ip[ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")
    requests_per_ip[ip].append(now)

# ---------------------------
# Chat endpoint
# ---------------------------
@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    client_ip = request.client.host
    check_rate_limit(client_ip)

    user_message = req.message.strip()
    page_name = req.page.strip() or "/"
    answers = []
    combined_contexts = []

    if knowledge_embeddings is not None and user_message:
        query_embedding = get_embedder().encode(user_message, convert_to_tensor=True)
        top_hits = util.semantic_search(query_embedding, knowledge_embeddings, top_k=5)[0]

        for hit in top_hits:
            if hit["score"] > 0.4:
                context_text = knowledge_chunks[hit["corpus_id"]]
                context_name = knowledge_files[hit["corpus_id"]].replace(".txt", "")
                combined_contexts.append(f"Page: {context_name}\n{context_text}")

        if combined_contexts:
            prompt = (
                "You are a helpful assistant. Answer the question using the following context:\n\n"
                + "\n\n".join(combined_contexts)
                + f"\n\nQuestion: {user_message}\nAnswer:"
            )
            try:
                qa_pipeline = get_qa_pipeline()
                result = qa_pipeline(prompt, max_length=512, do_sample=True)[0]
                answer_text = result["generated_text"].strip()

                if answer_text:
             # Split numbered/bulleted items into separate lines

                    lines = []
                    for line in answer_text.split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                         # Split numbered/dash items
                        sublines = re.split(r'(\d+\.\s)', line)
                        

                        temp = ""
                        for sub in sublines:
                            sub = sub.strip()
                            if not sub:
                                continue
                            if re.match(r'\d+\.', sub):
                                if temp:
                                    lines.append(temp.strip())
                                temp = sub
                            else:
                                temp += " " + sub
                        if temp:
                            lines.append(temp.strip())

                    formatted_answer = "\n".join(lines)
                    answers.append(f"You are currently on the page: {page_name}\n{formatted_answer}")

            except Exception as e:
                print("Flan-T5 error:", e)

    # Only fallback to joke if no top hits / context exists

    if not combined_contexts or not answers:
        joke = random.choice(jokes) if jokes else "Hmm, I don’t know… yet!"
        answers.append(f"You are currently on the page: {page_name}\n{joke}")

    return {"reply": "\n\n".join(answers)}
