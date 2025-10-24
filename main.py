from scripts.routes.price_prediction_router import price_prediction_router
import uvicorn
from fastapi import FastAPI
app = FastAPI(
    title = "Real Estate Price Prediction API",
    description="Predict real estate prices using input features",
    version="1.0.0"
)

app.include_router(price_prediction_router)

@app.get("/health")
def health():
    return {"message": "API is running"}
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)