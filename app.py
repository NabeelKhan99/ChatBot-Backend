import os
import random
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import re

os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf"
os.environ["HF_HOME"] = "/tmp/hf"

app = FastAPI()

# CORS for frontend
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

# Globals (lazy loaded later)
qa_pipeline = None
embedder = None

def get_qa_pipeline():
    global qa_pipeline
    if qa_pipeline is None:
        model_name = "google/flan-t5-small"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir="/tmp/hf")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/tmp/hf")
        qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return qa_pipeline

def get_embedder():
    global embedder
    if embedder is None:
        embedder = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="/tmp/hf")
    return embedder

# Load knowledge chunks
knowledge_chunks = []
knowledge_files = []
if os.path.exists("knowledge_base"):
    for fname in os.listdir("knowledge_base"):
        if fname.endswith(".txt"):
            text = open(os.path.join("knowledge_base", fname), encoding="utf-8").read()
            text = re.sub(r'(\d+)\s', r'\n\1- ', text)
            knowledge_chunks.append(text.strip())
            knowledge_files.append(fname)

knowledge_embeddings = (
    get_embedder().encode(knowledge_chunks, convert_to_tensor=True) if knowledge_chunks else None
)

# Load jokes
jokes = []
if os.path.exists("jokes.txt"):
    with open("jokes.txt", encoding="utf-8") as f:
        jokes = [line.strip() for line in f.readlines() if line.strip()]

@app.on_event("startup")
async def preload_models():
    """
    Warmup: download Hugging Face models into /tmp
    so first user request is not slow.
    """
    _ = get_qa_pipeline()
    _ = get_embedder()

@app.post("/chat")
async def chat(req: ChatRequest):
    user_message = req.message.strip()
    page_name = req.page.strip() or "/"
    answers = []

    if knowledge_embeddings is not None and user_message:
        query_embedding = get_embedder().encode(user_message, convert_to_tensor=True)
        top_hits = util.semantic_search(query_embedding, knowledge_embeddings, top_k=5)[0]

        combined_contexts = []
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
                result = get_qa_pipeline()(prompt, max_length=512, do_sample=True)[0]
                answer_text = result["generated_text"].strip()

                if answer_text:
                    lines = []
                    for line in answer_text.split("\n"):
                        line = line.strip()
                        if not line:
                            continue
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

    if not combined_contexts or not answers:
        joke = random.choice(jokes) if jokes else "Hmm, I don’t know… yet!"
        answers.append(f"You are currently on the page: {page_name}\n{joke}")

    return {"reply": "\n\n".join(answers)}
