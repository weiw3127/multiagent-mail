from typing import List, Union
import numpy as np
import torch
import soundfile as sf 
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, HubertForSequenceClassification
from app.schema import AgentOutput

#Option2: "HyperMoon/wav2vec2-base-960h-finetuned-deepfake"
#Option3: "abhishtagatya/hubert-base-960h-asv19-deepfake" (x)
MODEL_ID = "abhishtagatya/hubert-base-960h-itw-deepfake"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = AutoConfig.from_pretrained(MODEL_ID)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
model = HubertForSequenceClassification.from_pretrained(MODEL_ID, config=config).to(device).eval()


TARGET_SR = 16000

#the model was fine-tuned on 16 kHz mono waveforms, so need to normalize every clip 
def _load_audio_to_16k(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1) 
    if sr != TARGET_SR:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        except Exception:
            raise RuntimeError(f"Install librosa to resample {sr}->16k (pip install librosa).")
    return audio

@torch.no_grad()
def _score_one(wave_16k: np.ndarray) -> float:
    inputs = feature_extractor(wave_16k, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    for k in inputs: inputs[k] = inputs[k].to(device)
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    return float(probs[1].item())

def run(audio_paths: List[str]) -> AgentOutput:
    if not audio_paths:
        return AgentOutput(agent="audio_deepfake", score=0.0, reasons=["No audio provided"], features={})

    reasons, feats = [], {"clips": []}
    worst = 0.0
    for p in audio_paths[:5]:
        w = _load_audio_to_16k(p)
        s = _score_one(w)
        worst = max(worst, s)
        feats["clips"].append({"path": p, "deepfake_prob": s})
        if s >= 0.7:
            reasons.append(f"Audio deepfake model flagged {p} (p={s:.2f})")

    return AgentOutput(agent="audio_deepfake", score=worst, reasons=sorted(set(reasons)), features=feats)
