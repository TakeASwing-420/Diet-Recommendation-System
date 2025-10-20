from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List,Optional
import pandas as pd
from model import recommend,output_recommended_recipes


dataset=pd.read_csv('../Data/dataset.csv',compression='gzip')

app = FastAPI()


class Params(BaseModel):
    n_neighbors:int=5
    return_distance:bool=False

class PredictionIn(BaseModel):
    nutrition_input: List[float] = Field(..., min_length=9, max_length=9)
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
    params_dict = prediction_input.params.dict() if prediction_input.params is not None else {}
    recommendation_dataframe=recommend(dataset,prediction_input.nutrition_input,prediction_input.ingredients,params_dict)
    output=output_recommended_recipes(recommendation_dataframe)
    if output is None:
        return {"output":None}
    else:
        return {"output":output}

