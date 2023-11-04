from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from typing import Literal, Annotated
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
model_name = "model_furniture.pkl"
model = pickle.load(open(f"{dir_path}\\{model_name}", 'rb'))
app_furniture.mount("/static", StaticFiles(directory=f"{dir_path}\\static"), name="static")
templates = Jinja2Templates(directory=f"{dir_path}\\templates")


@app_furniture.get("/", response_class=HTMLResponse)
def home(request: Request):
 return templates.TemplateResponse("index_furniture.html", {"request": request, "prediction_text": ""})


@app_furniture.post('/predict', response_class=HTMLResponse)
async def predict(request: Request,
                  category: Annotated[int, Form()],
                  sellable_online: Annotated[bool, Form()],
                  other_colors: Annotated[bool, Form()],
                  depth: Annotated[float, Form()],
                  height: Annotated[float, Form()],
                  width: Annotated[float, Form()]):

    features = furniture(category=category,
                         sellable_online=sellable_online,
                         other_colors=other_colors,
                         depth=depth,
                         height=height,
                         width=width)
    pred = str(model.predict([[features.category,
                                            features.sellable_online,
                                            features.other_colors,
                                            features.depth,
                                            features.height,
                                            features.width]])[0])
    return templates.TemplateResponse("index_furniture.html", {"request": request, "prediction_text": f"prediction value : {pred}"})


if __name__ == "__main__":
 uvicorn.run("app_furniture_html:app_furniture", host="0.0.0.0", port=8080, reload=True)