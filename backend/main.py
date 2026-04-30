from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from full_process_v1 import PhishingExplainer, analyze_url

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST"])

explainer = PhishingExplainer()

class AnalyzeRequest(BaseModel):
    url: str

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    result = analyze_url(req.url, explainer, verbose=False)
    return {
        "prediction":  result["prediction"],
        "confidence":  result["confidence"],
        "explanation": result["explanation"][:10]  # top 5 pre extension
    }
