"""
Point d'entrée du serveur FastAPI — expose le pipeline RAG en HTTP.
"""

from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="RAG API", description="API pour interroger la base de connaissances RAG")

app.include_router(router)


@app.get("/health")
def health_check():
    return {"status": "ok"}