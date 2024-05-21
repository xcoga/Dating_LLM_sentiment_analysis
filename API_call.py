from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
import uvicorn
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from db_interactions import test_retrieve_image
from text_extraction import extract_text

app = FastAPI()

class User(BaseModel):
    name: str
    email: str
    password: str

class Image(BaseModel):
    image_id: str
    collection: str



@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/get_image_text/")
async def get_image_text(user: User, image: Image):
    # Use the User's credentials and Image details to find the image in the database.
    # Retrieve image from database.
    print(test_retrieve_image())
    img_path = "/home/datasets/data/images/val/Screenshot_20240329_000522_Telegram.jpg"

    # Run the text extraction
    messages = extract_text(img_path)






    return {"message": "extracted successfully.", "text": messages}


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
