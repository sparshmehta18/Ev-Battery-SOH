from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import predict

app = FastAPI(
    title="EV Battery Health, RUL & Resale Value Prediction API",
    description="Predicts EV battery SOH, Remaining Useful Life, degradation "
                "forecast, health score and an AI-estimated resale valuation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, tags=["prediction"])

@app.get("/")
async def root():
    return {"status": "ok", "service": "ev-battery-rul-api"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
