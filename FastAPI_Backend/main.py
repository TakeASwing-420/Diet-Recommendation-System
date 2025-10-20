from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List,Optional
import pandas as pd
import os
import logging
import time
from functools import lru_cache
from model import recommend, output_recommended_recipes, pretrain_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset configuration from environment variables
DATA_PATH = os.environ.get("DATASET_PATH", "../Data/dataset.csv")
DATA_COMPRESSION = os.environ.get("DATASET_COMPRESSION", "gzip")

@lru_cache()
def load_dataset(path: str = DATA_PATH, compression: str = DATA_COMPRESSION) -> pd.DataFrame:
    """
    Load the dataset with error handling and logging.
    Uses lru_cache to ensure the dataset is only loaded once.
    """
    try:
        logger.info(f"Loading dataset from {path} with compression={compression}")
        df = pd.read_csv(path, compression=compression, low_memory=False)
        logger.info(f"Successfully loaded dataset: {len(df)} rows, {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        logger.error(f"Dataset file not found at {path}")
        raise
    except Exception as e:
        logger.exception(f"Failed to load dataset from {path}: {e}")
        raise

# Global variables
dataset = None
pretrained_models = None

# Initialize FastAPI app
app = FastAPI()

@app.on_event("startup")
def startup_event():
    """Load dataset and pre-train models on application startup"""
    global dataset, pretrained_models
    logger.info("=== Startup event triggered ===")
    
    try:
        # Load dataset
        logger.info("Attempting to load dataset...")
        start_time = time.time()
        dataset = load_dataset()
        load_time = time.time() - start_time
        logger.info(f"=== Dataset loaded successfully: {dataset.shape} rows in {load_time:.2f}s ===")
        
        # Pre-train models (TF-IDF + NearestNeighbors on full dataset)
        logger.info("Pre-training TF-IDF and NearestNeighbors models on full dataset...")
        start_time = time.time()
        pretrained_models = pretrain_model(dataset)
        train_time = time.time() - start_time
        logger.info(f"=== Models pre-trained successfully in {train_time:.2f}s ===")
        logger.info(f"    - Feature dimensions: {pretrained_models['feature_dim']}")
        logger.info(f"    - TF-IDF vocabulary size: {pretrained_models['tfidf_vocab_size']}")
        logger.info("=== Startup complete - Ready to serve requests ===")
        
    except Exception as e:
        logger.exception(f"CRITICAL: Failed to initialize application: {e}")
        raise RuntimeError(f"Cannot start application: {e}")

@app.on_event("shutdown")
def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("=== Application shutdown ===")


class Params(BaseModel):
    n_neighbors:int=5
    return_distance:bool=False

class PredictionIn(BaseModel):
    nutrition_input: List[float] = Field(..., min_items=9, max_items=9)
    ingredients: List[str] = Field(default_factory=list)
    params: Optional[Params] = None


class Recipe(BaseModel):
    Name:str
    CookTime:str
    PrepTime:str
    TotalTime:str
    RecipeIngredientParts:list[str]
    Calories:float
    FatContent:float
    SaturatedFatContent:float
    CholesterolContent:float
    SodiumContent:float
    CarbohydrateContent:float
    FiberContent:float
    SugarContent:float
    ProteinContent:float
    RecipeInstructions:list[str]

class PredictionOut(BaseModel):
    output: Optional[List[Recipe]] = None


@app.get("/")
def home():
    return {"health_check": "OK"}


@app.post("/predict/",response_model=PredictionOut)
def update_item(prediction_input:PredictionIn):
    """
    Generate recipe recommendations based on nutritional input and optional ingredient filters.
    Uses pre-trained models for fast inference.
    """
    start_time = time.time()
    params_dict = prediction_input.params.dict() if prediction_input.params is not None else {}
    
    # Use pre-trained models for fast recommendations
    recommendation_dataframe = recommend(
        dataset,
        prediction_input.nutrition_input,
        prediction_input.ingredients,
        params_dict,
        pretrained_models=pretrained_models  # Pass pre-trained models
    )
    
    output = output_recommended_recipes(recommendation_dataframe)
    
    inference_time = time.time() - start_time
    logger.info(f"Recommendation generated in {inference_time:.3f}s (ingredients filter: {len(prediction_input.ingredients)})")
    
    if output is None:
        return {"output":None}
    else:
        return {"output":output}

