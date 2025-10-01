from transformers import pipeline 
from app.schemas import AgentOutput, Span

url_clf = pipeline("text-classification", model="Eason918/malicious-url-detector-v2")

def run(urls):
    reasons, features, worst = [], {"urls": []}, 0.0
    for u in urls[:10]:
        vote = url_clf(u, truncation=True)[0]
        score = float(vote["score"]) if vote["label"].lower().startswith("mal") else 1 - float(vote["score"])
        worst = max(worst, score)
        features["urls"].append({"url": u, "clf": vote})
        if score > 0.7: reasons.append(f"URL model red flag {u}")
    return AgentOutput(agent="url", score=worst, reasons=sorted(set(reasons)), features=features)

