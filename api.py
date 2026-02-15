from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_service import get_model_service
import uvicorn

app = FastAPI(title="SentiMind: Mental Health & Sentiment API")

class PredictionRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: PredictionRequest):
    service = get_model_service()
    result = service.predict(request.text)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return {
        "text": result["text"],
        "sentiment": result["overall_sentiment"],
        "mental_health_indicator": result["mental_health_indicator"],
        "sentiment_score": result["sentiment_score"],
        "confidence": result["top_prediction"]["confidence"],
        "all_scores": result["predictions"]
    }

@app.get("/health")
async def health():
    service = get_model_service()
    if service.model is None:
        return {"status": "model_not_loaded"}
    return {"status": "ready"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
