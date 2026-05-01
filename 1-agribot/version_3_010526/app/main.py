# app/main.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from inference import load_model, generate
#from rag import load_kb, retrieve, build_rag_prompt
from guardrails import (is_farming_related, is_valid_response,
                        OFF_TOPIC_MSG, FALLBACK_MSG)

app = FastAPI(title="AgriBot")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── Load models once at startup ──────────────────────────────────────────────
print("Loading models...")
base_model,       base_tok       = load_model("google/flan-t5-base")
finetuned_model,  finetuned_tok  = load_model("fabienyah321/agribot-flan-t5")
#faiss_index, kb_docs             = load_kb("faiss_index.bin", "kb_docs.pkl")
print("All models loaded!")

class Query(BaseModel):
    question: str
    mode: str  # "base" | "finetuned" | "finetuned_rag" | "finetuned_guard" | "finetuned_rag_guard"

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    #return templates.TemplateResponse("index.html", {"request": request})
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/ask")
async def ask_endpoint(query: Query):
    q = query.question.strip()

    if query.mode == "base":
        answer = generate(q, base_model, base_tok)

    elif query.mode == "finetuned":
        answer = generate(q, finetuned_model, finetuned_tok)
    
    elif query.mode == 'finetuned_guard':
        if not is_farming_related(q) or not is_valid_response(q):
            answer = OFF_TOPIC_MSG
        else:
            answer = generate(q, finetuned_model, finetuned_tok)
            
    # elif query.mode == "finetuned_rag":
    #     chunks = retrieve(q, faiss_index, kb_docs)
    #     if chunks:
    #         prompt = build_rag_prompt(q, chunks)
    #         answer = generate(prompt, finetuned_model, finetuned_tok)
    #     else:
    #         answer = FALLBACK_MSG

    # elif query.mode == "finetuned_rag_guard":
    #     if not is_farming_related(q):
    #         answer = OFF_TOPIC_MSG
    #     else:
    #         chunks = retrieve(q, faiss_index, kb_docs)
    #         if not chunks:
    #             answer = FALLBACK_MSG
    #         else:
    #             prompt = build_rag_prompt(q, chunks)
    #             raw    = generate(prompt, finetuned_model, finetuned_tok)
    #             answer = raw if is_valid_response(raw) else f"Based on my knowledge base: {chunks[0]['text']}"
    else:
        answer = "Unknown mode."

    return JSONResponse({"answer": answer, "mode": query.mode})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)