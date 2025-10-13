import argparse
import json
import os
from pathlib import Path
from typing import List

# Import your orchestrator + request schema
try:
    from app.phone_orchestrator import PhoneOrchestrator
    from app.schema import AnalyzeCallRequest, FinalDecision
except Exception as e:
    raise SystemExit(
        "ERROR: Could not import app.phone_orchestrator / app.phone_schema. "
        "Run from your project root and ensure PYTHONPATH includes it.\n"
        f"{e}"
    )

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg"}

def is_audio(p: Path) -> bool:
    return p.suffix.lower() in AUDIO_EXTS

def main():
    ap = argparse.ArgumentParser(
        description="Deepfake check for one or more audio clips using PhoneOrchestrator."
    )
    ap.add_argument(
        "inputs",
        nargs="+",
        help="One or more paths (files or directories). Directories will be scanned recursively for audio files.",
    )
    ap.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print JSON decisions."
    )
    args = ap.parse_args()

    # Collect audio paths
    paths: List[str] = []
    for raw in args.inputs:
        p = Path(raw)
        if not p.exists():
            print(json.dumps({"path": raw, "error": "not_found"}))
            continue
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and is_audio(f):
                    paths.append(str(f))
        else:
            if is_audio(p):
                paths.append(str(p))
            else:
                print(json.dumps({"path": str(p), "error": "not_audio"}))

    if not paths:
        raise SystemExit("No valid audio files found in the provided inputs.")

    # Initialize orchestrator once
    orch = PhoneOrchestrator()

    # Score each file independently
    for clip in paths:
        try:
            req = AnalyzeCallRequest(audio_paths=[clip])
            decision: FinalDecision = orch.analyze(req)
            out = {
                "path": clip,
                "risk_label": decision.risk_label,
                "risk_prob": decision.risk_prob,
                "top_reasons": decision.top_reasons,
                "agent_outputs": [ao.model_dump() for ao in decision.agent_outputs],
            }
            print(json.dumps(out, indent=2 if args.pretty else None))
        except Exception as e:
            print(json.dumps({"path": clip, "error": str(e)}))

if __name__ == "__main__":
    main()
