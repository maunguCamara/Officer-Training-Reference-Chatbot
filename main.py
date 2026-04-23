"""
TrainingBot Backend — FastAPI entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import auth, ingest, chat, whatsapp

app = FastAPI(
    title="TrainingBot API",
    description="AI-powered training assistant with RAG and citations",
    version="1.0.0",
)

# Allow React Native app to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your app domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(ingest.router)
app.include_router(chat.router)
app.include_router(whatsapp.router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "TrainingBot API"}
