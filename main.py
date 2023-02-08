from starlette.responses import RedirectResponse
from uvicorn import run as app_run
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Form
from pydantic import BaseModel
import pickle
import json
import sys
import spacy
import gensim.downloader as api
from logger import logging
from exception import FakeNewsException

APP_HOST = "0.0.0.0"
APP_PORT = 8000

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")
# Creating base model for input parameters
class input_model(BaseModel):
    Text:str 
     

wc = api.load("word2vec-google-news-300")

nlp = spacy.load("en_core_web_lg")
def preprocess_and_vectorize(text):
    try:
        logging.info("Starting preprocessing and vectorizing")
        doc = nlp(text)
        filtered_token = []
        for token in doc:
            if token.is_stop or token.is_punct:
                continue
            filtered_token.append(token.lemma_)
            
        return wc.get_mean_vector(filtered_token)

    except Exception as e:
        raise FakeNewsException(e, sys)
# Loading model
model = pickle.load(open('fake_news_classifier.sav', 'rb'))

@app.post("/fake_news_classifier")
async def fake_news_classifier(Text:str = Form()):
    try:
        logging.info('fake_news_classifier called')   

        vector = [preprocess_and_vectorize(Text)]
        prediction = model.predict(vector)     


        if prediction[0] == 0:
            return "This News is likely to be Fake"

        if prediction[0]== 1:
            return "This News is likely to be Real"
    
    except Exception as e:
        raise FakeNewsException(e, sys)


if __name__=="__main__":
    
    app_run(app, host=APP_HOST, port=APP_PORT)