from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Cookie
from pydantic import BaseModel

from rag_pipeline.data_loader import CocktailLoader
from rag_pipeline.vector_store import CocktailVectorStore
from rag_pipeline.llm_interface import LocalLLM
from rag_pipeline.user_memory import UserMemory
from rag_pipeline.rag_engine import RAGEngine

from huggingface_hub import hf_hub_download
import os

from uuid import uuid4

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ====== CHECK & DOWNLOAD MODEL ======
FILENAME = "tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"
REPO = "stkrk/tinyllama-gguf"
MODEL_PATH = os.path.join("models", FILENAME)

if not os.path.exists(MODEL_PATH):
    print(f"Model not found, downloading from {REPO}...")
    hf_hub_download(
        repo_id=REPO,
        filename=FILENAME,
        cache_dir="models",
        local_dir="models",
        local_dir_use_symlinks=False
    )
    print("Model downloaded.")
else:
    print("Model already exists.")

# ===== 1. Main pipeline =====
print("Initializing RAG pipeline...")
loader = CocktailLoader("data/cocktails.csv")
cocktails = loader.load()

vector_store = CocktailVectorStore(cocktails)
llm = LocalLLM(MODEL_PATH)
memory = UserMemory()
engine = RAGEngine(vector_store, llm, memory)

print("RAG engine is ready!")


# ===== 2. HTML-chat =====
chat_histories = {}  # session_id -> list of chat messages

@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request, session_id: str = Cookie(default=None)):
    if not session_id:
        session_id = str(uuid4())

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    response = templates.TemplateResponse("chat.html", {
        "request": request,
        "chat_history": chat_histories[session_id]
    })

    response.set_cookie(key="session_id", value=session_id)
    return response


@app.post("/ask", response_class=HTMLResponse)
async def ask(request: Request, message: str = Form(...), session_id: str = Cookie(default=None)):
    if not session_id:
        session_id = str(uuid4())

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    response = engine.run(message)
    chat_histories[session_id].append({"user": message, "bot": response})

    html = templates.TemplateResponse("chat.html", {
        "request": request,
        "chat_history": chat_histories[session_id]
    })

    html.set_cookie(key="session_id", value=session_id)
    return html


@app.get("/static/favicon.png", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.png")

# ===== 3. REST API endpoint =====

class Query(BaseModel):
    message: str

@app.post("/api/ask")
async def api_ask(query: Query):
    response = engine.run(query.message)
    return {"answer": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("run_fast_api:app", host="0.0.0.0", port=7860)

