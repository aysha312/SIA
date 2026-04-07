from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from transformers import pipeline
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

nlp = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title=" Sentiment Analysis API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

auth_scheme = HTTPBearer()
API_TOKEN = "student-access-2026"

class TextInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Welcome to the Enhanced Sentiment Analysis API! Use /analyze for NLP tasks.",
        "version": "1.1.0"
    }

@app.post("/analyze")
@limiter.limit("5/minute")
def analyze_sentiment(
    request: Request,
    payload: TextInput,
    token: HTTPAuthorizationCredentials = Depends(auth_scheme)
):
    if token.credentials != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid Security Token")
    
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    result = nlp(text)
    sentiment = result[0]['label']
    score = round(result[0]['score'], 4)
    
    if sentiment == "POSITIVE":
        message = f" Great! That sounds positive. Confidence: {score}"
    else:
        message = f" Hmm, that seems negative. Confidence: {score}"
    
    return {
        "status": "success",
        "input_text": text,
        "sentiment": sentiment,
        "confidence_score": score,
        "message": message
    }
