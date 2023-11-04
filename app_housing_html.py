from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from typing import Annotated
app_housing = FastAPI()
class housing(BaseModel):
    surface:float
    rooms:int
    yard:bool
    pool:bool
    floors:int
    new:float
import pickle
#we are loading the model using pickle
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
model_name = "model_housing.pkl"
model = pickle.load(open(f"{dir_path}\\{model_name}", 'rb'))
app_housing.mount("/static", StaticFiles(directory=f"{dir_path}\\static"), name="static")
templates = Jinja2Templates(directory=f"{dir_path}\\templates")


@app_housing.get("/", response_class=HTMLResponse)
def home(request: Request):
 return templates.TemplateResponse("index_housing.html", {"request": request, "prediction_text": ""})


@app_housing.post('/predict', response_class=HTMLResponse)
async def predict(request: Request,
                  surface: Annotated[float, Form()],
                  rooms: Annotated[int, Form()],
                  yard: Annotated[bool, Form()],
                  pool: Annotated[bool, Form()],
                  floors: Annotated[int, Form()],
                  new: Annotated[bool, Form()]):

    features = housing(surface=surface,
                         rooms=rooms,
                         yard=yard,
                         pool=pool,
                         floors=floors,
                         new=new)
    pred = str(model.predict([[features.surface,
                                            features.rooms,
                                            features.yard,
                                            features.pool,
                                            features.floors,
                                            features.new]])[0])
    return templates.TemplateResponse("index_housing.html", {"request": request, "prediction_text": f"prediction value : {pred}"})


if __name__ == "__main__":
 uvicorn.run("app_housing_html:app_housing", host="0.0.0.0", port=8080, reload=True)