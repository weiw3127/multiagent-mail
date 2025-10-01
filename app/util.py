from app.schema import AgentOutput, FinalDecision
from typing import List, Tuple
from bs4 import BeautifulSoup
import re, tldextract 

URL_RE = re.compile(r'https?://[^\s)>\]"]+', re.I)

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for s in soup(["script", "style"]): s.decompose()
    return soup.get_text(" ", strip=True)

def extract_urls(html: str, text: str) -> List[str]: 
    candidates = set()
    if html: candidates |= set(URL_RE.findall(html))
    if text: candidates |= set(URL_RE.findall(text))
    return list(candidates)

def canonical_domain(url: str) -> str: 
    ext = tldextract.extract(url)
    return ".".join([p for p in [ext.domain, ext.suffix] if p])


def decide(outputs: List[AgentOutput]) -> FinalDecision:
        # simple average the available agent scores
        scores = [float(o.score) for o in outputs if o is not None]
        risk = sum(scores) / len(scores) if scores else 0.0
        risk = float(round(risk, 3))

        if risk >= 0.70:
            label = "HIGH"
        elif risk >= 0.40:
            label = "MEDIUM"
        else:
            label = "LOW"

        reasons = []
        for o in outputs:
            if o:
                reasons.extend(o.reasons)

        top_reasons = reasons[:5]

        return FinalDecision(
            risk_label=label,
            risk_prob=risk,
            top_reasons=top_reasons,
            agent_outputs=outputs
        )
    

