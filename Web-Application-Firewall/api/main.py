from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from waf.rules.engine import RuleEngine
from waf.ml.classifier import WAFClassifier

app = FastAPI(title="Web Application Firewall API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

_rule_engine = RuleEngine()
_ml_model: WAFClassifier | None = None

_MODEL_PATH = Path(os.getenv("WAF_MODEL_PATH", "models/waf_rf.pkl"))


@app.on_event("startup")
def _load_model() -> None:
    global _ml_model
    if _MODEL_PATH.exists():
        try:
            _ml_model = WAFClassifier.load(_MODEL_PATH)
        except Exception as e:
            print(f"[WAF] Could not load ML model from {_MODEL_PATH}: {e}")
    else:
        print(f"[WAF] No model file at {_MODEL_PATH} — ML scoring disabled. "
              "Run scripts/train.py to generate one.")


class RequestPayload(BaseModel):
    method:  str  = "GET"
    url:     str
    query:   str  = ""
    body:    str  = ""
    headers: dict = {}


class Verdict(BaseModel):
    malicious:        bool
    confidence:       float | None   # ML probability if model loaded
    rule_score:       int
    primary_category: str | None
    categories:       list[str]
    rule_matches:     list[dict]
    ml_available:     bool


@app.post("/inspect", response_model=Verdict)
def inspect(payload: RequestPayload) -> Verdict:
    req = payload.model_dump()

    rule_verdict = _rule_engine.inspect(req)

    ml_prob: float | None = None
    if _ml_model is not None:
        try:
            _, ml_prob = _ml_model.predict(req)
        except Exception:
            ml_prob = None

    # Combined decision: malicious if rules fire OR ML probability > 0.5
    malicious = rule_verdict.malicious
    if ml_prob is not None and ml_prob > 0.5:
        malicious = True

    matches = [
        {
            "category":     m.rule.category,
            "severity":     m.rule.severity,
            "description":  m.rule.description,
            "field":        m.field,
            "matched_text": m.matched_text,
        }
        for m in rule_verdict.matches
    ]

    return Verdict(
        malicious=malicious,
        confidence=ml_prob,
        rule_score=rule_verdict.score,
        primary_category=rule_verdict.primary_category,
        categories=sorted(rule_verdict.categories),
        rule_matches=matches,
        ml_available=_ml_model is not None,
    )


@app.get("/health")
def health() -> dict:
    return {
        "status":       "ok",
        "ml_model":     _ml_model.model_type if _ml_model else None,
        "rules_loaded": True,
    }
