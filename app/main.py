from pathlib import Path
from typing import Annotated, List

import getpass
import os
import shutil
import sys
import tempfile

from fastapi import APIRouter, Depends, FastAPI, HTTPException, UploadFile, File

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.email_orchestrator import EmailOrchestrator
from app.schema import AnalyzeCallRequest, AnalyzeEmailRequest, FinalDecision

try:
    from app.phone_orchestrator import PhoneOrchestrator
    _phone_orchestrator = PhoneOrchestrator()
except Exception as _e:
    _phone_orchestrator = None
    print("[WARN] PhoneOrchestrator could not be initialized:", _e)

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")


app = FastAPI(title="mailMAS")


def get_email_orchestrator() -> EmailOrchestrator:
    return EmailOrchestrator()


router = APIRouter()

EmailMAS = Annotated[EmailOrchestrator, Depends(get_email_orchestrator)]

@router.get("/health")
async def health_check() -> dict:
    return {"ok": True}


@router.post("/email/analyze", response_model=FinalDecision)
async def analyze_email(model: EmailMAS, request: AnalyzeEmailRequest) -> FinalDecision:
    decision = model.analyze(request)
    if decision is None:
        raise HTTPException(status_code=500, detail="Error analyzing email")
    return decision


@app.post("/phone/analyze-audio")
async def analyze_phone_audio(files: List[UploadFile] = File(...)):
    
    if _phone_orchestrator is None:
        return {"error": "PhoneOrchestrator unavailable. Check dependencies and imports."}

    tmpdir = tempfile.mkdtemp(prefix="phone-audio-")
    paths = []
    try:
        for f in files:
            out_path = os.path.join(tmpdir, f.filename or "clip")
            with open(out_path, "wb") as w:
                w.write(await f.read())
            paths.append(out_path)

        req = AnalyzeCallRequest(audio_paths=paths)
        decision = _phone_orchestrator.analyze(req)
        return decision.model_dump()
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


app.include_router(router)
