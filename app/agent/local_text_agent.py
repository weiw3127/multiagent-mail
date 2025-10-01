from transformers import pipeline
from app.schema import AgentOutput 

clf = pipeline("text-classification", model="dima806/phishing-email-detection") 

def run(subject, body_text):
    text = f"{subject}\n{body_text or ''}"
    pred = clf(text, truncation=True, max_length = 512)[0]
    label = pred["label"].lower()
    score = float(pred["score"]) if label.startswith("phish") else 1 - float(pred["score"])
    reasons = [f"Classifier vote: {label}"]
    return AgentOutput(agent="text", score=score, reasons=reasons, features={})