"""
Endpoints exposés par l'API RAG.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from src.main import ask

router = APIRouter()


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5


class AnswerResponse(BaseModel):
    answer: str


@router.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    Reçoit une question, retourne une réponse ancrée dans les documents RAG.
    C'est cet endpoint que ton modèle/bot DeepSeek externe doit appeler.
    """
    answer = ask(request.question, top_k=request.top_k)
    return AnswerResponse(answer=answer)