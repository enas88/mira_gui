import os
import shutil
import numpy as np
import pandas as pd

from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sentence_transformers
from sentence_transformers import SentenceTransformer, util
from semantic_matching.code.semantic_matching_tables import *

import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


#####################################################################
# CSV Upload API

# Directory to store uploaded CSV files
# UPLOAD_DIR = "uploads"
UPLOAD_DIR = "semantic_matching/data/"

# A list to store CSV records
csv_records = []
[csv_records.append({"name": filename}) for filename in os.listdir(UPLOAD_DIR) if filename.endswith('.csv')]

# Route to list CSV records
@app.get("/csv/")
async def list_csv_records():
    return csv_records

# Route to upload a CSV file
@app.post("/csv/")
async def upload_csv_file(file: UploadFile):
    # Ensure the uploads directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    csv_records.append({"name": file.filename})

    create_embeddings(file_path, '')

    return JSONResponse(content={"message": "CSV file uploaded successfully"}, status_code=201)


#####################################################################
# Search API #

# Create query object class
class Query(BaseModel):
    query_text: str

# Define input data and create embeddings
db = np.array(['Data Scientist','Machine Learning Engineer','Data Analyst','Software Developer','Front End Developer','Back End Developer','Mathematician','Physicist'])
encodings = model.encode(db)

@app.post("/exhaustive_search/")
async def exhaustive_search(query: Query):

    # Get all csv files:
    base_path = 'semantic_matching/data/'
    csv_files = [base_path+file for file in os.listdir(base_path) if file.endswith('.csv')]

    k=20
    top_k_results = batch_semantic_matching(query.query_text, csv_files, k).drop('Embeddings', axis=1)

    return Response(top_k_results.to_json(orient="records"), media_type="application/json")


#####################################################################

# persons = []

# class Person(BaseModel):
#     first_name: str
#     last_name: str
#     age: int

# @app.get("/")
# async def get_form():
#     # return FileResponse("templates/form.html")
#     return FileResponse("templates/index.html")

# @app.post("/person/")
# async def add_person(person: Person):
#     persons.append(person.model_dump())
#     return {"message": "Person added successfully"}

# @app.get("/person/")
# async def list_persons():
#     return persons

# @app.get("/person/{index}/")
# async def get_person(index: int):
#     return persons[index]

# @app.delete("/person/{index}/")
# async def delete_person(index: int):
#     if 0 <= index < len(persons):
#         deleted_person = persons.pop(index)
#         return {"message": "Person deleted successfully", "deleted_person": deleted_person}
#     else:
#         return {"error": "Invalid index"}

# @app.put("/person/{index}/")
# async def update_person(index: int, person: Person):
#     if 0 <= index < len(persons):
#         persons[index] = person.model_dump()
#         return {"message": "Person updated successfully"}
#     else:
#         return {"error": "Invalid index"}
    

