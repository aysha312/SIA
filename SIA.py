from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from transformers import pipeline
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# 1. Initialize local NLP Engine
nlp = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# 2. Setup Rate Limiting
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="NLP Sentiment API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 3. Security Configuration
auth_scheme = HTTPBearer()
API_TOKEN = "student-access-2026"

# --- NEW: GET Method (Status Check) ---
@app.get("/")
def root():
    """
    Public endpoint to verify the API is running.
    No authentication required.
    """
    return {
        "status": "online",
        "message": "Welcome to the Sentiment Analysis API. Use /analyze for NLP tasks.",
        "version": "1.0.0"
    }

# --- REQUIRED: POST Method (Analysis) ---
@app.post("/analyze")
@limiter.limit("5/minute")
def analyze_sentiment(request: Request, text: str, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    """
    Protected endpoint for Sentiment Analysis.
    Requires Bearer Token: cadet-access-2026
    """
    if token.credentials != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid Security Token")
    
    result = nlp(text)
    
    return {
        "status": "success",
        "input_text": text,
        "label": result[0]['label'],
        "confidence_score": round(result[0]['score'], 4)
    }