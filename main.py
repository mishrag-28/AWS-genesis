# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 20:45:51 2024

@author: gm205
"""
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
#from Text import Text
from modelfile import NERModel
import uvicorn
from pydantic import BaseModel
# Load the NER module
ner = NERModel()

# User's input text is sent as a request body. The structure of this body can be defined by extending Pydantic's BaseModel
class Text(BaseModel):
    text: str
# The model above declares a JSON object (or Python dict) like:
# {
#     "text": "user's input text"
# }

# Initialize the FastAPI application
app = FastAPI()

# Allow Cross-Origin Resource Sharing (CORS) requests to from any host so that the JavaScript in the extension can communicate with the server
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# API root
@app.get("/")
def get_root():
    return "This is the RESTful API for your NER application"

# POST endpoint with path '/predict'
@app.post("/predict")
async def predict_entities(text: Text):
    entities = ner.predict_entities(text.text)

    return {
        "original": text.text,
        "entities": entities
    }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', 
                port=8080, 
                ssl_keyfile="/etc/letsencrypt/archive/ec2instance.ethix4ai.com/privkey1.pem", 
                ssl_certfile="/etc/letsencrypt/archive/ec2instance.ethix4ai.com/fullchain1.pem")


    # uvicorn.run(app, host='0.0.0.0', port=8080)
    # # uvicorn.run(app, host='0.0.0.0', port=6000)
#uvicorn main:app --reload


        # "default_popup": "popup.html"
