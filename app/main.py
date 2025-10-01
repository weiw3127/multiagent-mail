from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.email_orchestrator import EmailOrchestrator
from app.schema import AnalyzeEmailRequest, FinalDecision

import getpass
import os

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")


app = FastAPI(title="mailMAS")


def get_email_orchestrator() -> EmailOrchestrator:
    return EmailOrchestrator()


router = APIRouter()

EmailMAS = Annotated[EmailOrchestrator, Depends(get_email_orchestrator)]


@router.post("/email/analyze", response_model=FinalDecision)
async def analyze_email(model: EmailMAS, request: AnalyzeEmailRequest) -> FinalDecision:
    decision = model.analyze(request)
    if decision is None:
        raise HTTPException(status_code=500, detail="Error analyzing email")
    return decision


@router.get("/health")
async def health_check() -> dict:
    return {"ok": True}


app.include_router(router)
