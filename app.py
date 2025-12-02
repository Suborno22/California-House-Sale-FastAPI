from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import pandas as pd
from typing import List
import sys
from io import BytesIO
from huggingface_hub import hf_hub_download

# CRITICAL: Define these functions BEFORE loading the model!
def column_ratio(X):
    """This needs to be at module level for pickle to find it"""
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    """This needs to be at module level for pickle to find it"""
    return ["ratio"]

# PICKLE FIX: Add these functions to __mp_main__ module
if '__mp_main__' not in sys.modules:
    sys.modules['__mp_main__'] = sys.modules[__name__]
sys.modules['__mp_main__'].column_ratio = column_ratio
sys.modules['__mp_main__'].ratio_name = ratio_name

# Your custom transformer - must be defined before model loading!
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters, gamma, random_state):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.gamma = gamma
    
    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self
    
    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} Similarity" for i in range(self.n_clusters)]

# Update the __mp_main__ reference
sys.modules['__mp_main__'].ClusterSimilarity = ClusterSimilarity

# Global variable for the model
model = None

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model
    global model
    try:
        
        # In the lifespan function:
        model_path = hf_hub_download(
            repo_id="Suborno99/california-housing-price-prediction-model",
            filename="california_housing_model.pkl"
        )
        model = joblib.load(model_path)
            
    except Exception as e:
        print(f"We got a problem loading the model: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        model = None
    
    yield  # Server is running
    
    # Shutdown: cleanup if needed
    print("Shuttin' down... !")

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

# Pydantic models
class HousingData(BaseModel):
    longitude: float = Field(..., description="Longitude coordinate")
    latitude: float = Field(..., description="Latitude coordinate")
    housing_median_age: float = Field(..., description="Median age of housing")
    total_rooms: float = Field(..., description="Total rooms")
    total_bedrooms: float = Field(..., description="Total bedrooms")
    population: float = Field(..., description="Population")
    households: float = Field(..., description="Number of households")
    median_income: float = Field(..., description="Median income")
    ocean_proximity: str = Field(..., description="Ocean proximity category")

    class Config:
        json_schema_extra = {
            "example": {
                "longitude": -122.42,
                "latitude": 37.80,
                "housing_median_age": 52.0,
                "total_rooms": 3321.0,
                "total_bedrooms": 1115.0,
                "population": 1576.0,
                "households": 1034.0,
                "median_income": 2.0987,
                "ocean_proximity": "NEAR BAY"
            }
        }

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
            detail="Model ain't ready! What am I gonna do? Check /health endpoint."
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

# Run with: uvicorn app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)