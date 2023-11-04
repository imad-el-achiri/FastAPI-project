from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import Literal
app_furniture = FastAPI()
class furniture(BaseModel):
 category:Literal[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
 sellable_online:bool
 other_colors:bool
 depth:float
 height:float
 width:float
import pickle
#we are loading the model using pickle
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
model_name = "model.pkl"
model = pickle.load(open(f"{dir_path}\\{model_name}", 'rb'))
@app_furniture.get("/")
def home():
 return {'ML model for Furniture prediction'}
@app_furniture.post('/make_predictions')
async def make_predictions(features: furniture):
    return({"prediction":str(model.predict([[features.category,
                                             features.sellable_online,
                                             features.other_colors,
                                             features.depth,
                                             features.height,
                                             features.width]])[0])})
if __name__ == "__main__":
 uvicorn.run("app_furniture:app_furniture", host="0.0.0.0", port=8080, reload=True)
