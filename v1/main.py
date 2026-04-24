"""
TrainingBot Backend — FastAPI entry point.
WhatsApp-only: no React Native, no Firebase auth.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import whatsapp, ingest

app = FastAPI(
    title="TrainingBot API",
    description="WhatsApp-based AI training assistant with RAG and citations",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(whatsapp.router)
app.include_router(ingest.router)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "TrainingBot API", "version": "2.0.0"}
