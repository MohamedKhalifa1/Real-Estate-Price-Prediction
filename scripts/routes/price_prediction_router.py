from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from scripts.input_schema import InputSchema
from scripts.price_prediction import PricePrediction
import numpy as np

price_prediction_router = APIRouter(prefix="/predict", tags=["Price Prediction"])

@price_prediction_router.post("/")
def predict_price(input_data: InputSchema):
    try:
        # Initialize prediction class
        price_prediction = PricePrediction(
            model_path="models/best_model.pkl",
            scaler_path="models/scaler.pkl",
            model_columns_path="models/model_columns.pkl",
            pca_path="models/pca.pkl"
        )

        # Get prediction
        predicted_price = price_prediction.predict_price(input_data.model_dump())

        # Convert numpy types to Python native types
        if isinstance(predicted_price, (np.generic, np.ndarray)):
            predicted_price = predicted_price.item() if np.ndim(predicted_price) == 0 else predicted_price.tolist()

        # Return clean JSON
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"price": f"{predicted_price} EGP"}
        )

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
