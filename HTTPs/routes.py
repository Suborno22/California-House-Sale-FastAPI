from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

def Routes(BaseModel):
    
        # Initialize FastAPI with lifespan
    app = FastAPI(
        title="California Housing Price Predictor",
        lifespan=lifespan
    )

    # CORS - so your React app don't get blocked
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify your domain!
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    class PredictionResponse(BaseModel):
        predictions: List[float]
        count: int

    @app.get("/")
    async def root():
        return {
            "message": "Ey! California Housing Price Predictor API is runnin'!",
            "model_loaded": model is not None,
            "endpoints": {
                "/predict": "POST - Make predictions (list of houses)",
                "/predict_single": "POST - Make prediction (single house)",
                "/health": "GET - Check if everything's good",
                "/docs": "GET - Interactive API documentation"
            }
        }

    @app.get("/health")
    async def health_check():
        if model is None:
            raise HTTPException(
                status_code=503, 
                detail="Model ain't loaded yet! Check the server logs to see what's wrong."
            )
        return {
            "status": "healthy", 
            "model_loaded": True,
            "message": "Everything's good! Ready to make predictions."
        }

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(data: List[HousingData]):
        """Predict prices for multiple houses"""
        if model is None:
            raise HTTPException(
                status_code=503, 
                detail="Model ain't ready!Check /health endpoint."
            )
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([item.model_dump() for item in data])
            
            # Make predictions
            predictions = model.predict(df)
            
            return PredictionResponse(
                predictions=predictions.tolist(),
                count=len(predictions)
            )
        
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=400, 
                detail=f"Prediction failed: {str(e)}"
            )

    @app.post("/predict_single")
    async def predict_single(data: HousingData):
        """Predict for a single house - easier for testing"""
        if model is None:
            raise HTTPException(
                status_code=503, 
                detail="Model ain't loaded! Check the server logs."
            )
        
        try:
            df = pd.DataFrame([data.model_dump()])
            prediction = model.predict(df)
            
            return {
                "prediction": float(prediction[0]),
                "prediction_dollars": float(prediction[0] * 100000),
                "input_data": data.model_dump()
            }
        
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=400, 
                detail=f"Prediction failed: {str(e)}"
            )
