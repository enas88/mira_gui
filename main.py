import os
import time
import shutil
import numpy as np
import pandas as pd
import json 

from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile, Response, HTTPException
from fastapi.responses import FileResponse, JSONResponse

import qdrant_client
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

from pydantic import BaseModel

from config import *

import sentence_transformers
from sentence_transformers import SentenceTransformer, util
from semantic_matching.code.semantic_matching_tables import *

import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


#####################################################################

@app.get("/")
async def get_form():
    return FileResponse("templates/dashboard.html")

@app.get("/exhaustive")
async def get_form():
    return FileResponse("templates/index.html")

@app.get("/ann")
async def get_form():
    return FileResponse("templates/ann.html")

@app.get("/upload")
async def get_form():
    return FileResponse("templates/upload.html")

@app.get("/view")
async def get_form():
    return FileResponse("templates/view.html")

#####################################################################
# CSV Upload API

class CSV:

    def read_csv(file_path : str):
        df = pd.read_csv(file_path)

        return df.to_dict('records')
    

@app.get("/csv/{file_name}")
async def list_csv_records(file_name: str):

    try:
        csv_data = CSV.read_csv(UPLOAD_DIR+file_name)
        return csv_data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="CSV file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Route to list CSV records
@app.get("/csv/")
async def list_all_csv_records():

    qdrant_is_running = qdrant_is_working("localhost", 6333)
    client = QdrantClient(host="localhost", port=6333)
    # A list to store CSV records
    csv_records = []
    [csv_records.append({"name": filename,
                        "file_size": str(round(os.path.getsize(UPLOAD_DIR+filename) / 1024, 2))+' KB',
                        "exists_in_db": "-" if not qdrant_is_running else "Yes" if count_table_occurencies(filename, client, COLLECTION_NAME)>0 else "No",
                        "created_timestamp":time.ctime(os.path.getctime(UPLOAD_DIR+filename)),
                        "modified_timestamp":time.ctime(os.path.getmtime(UPLOAD_DIR+filename))}) for filename in os.listdir(UPLOAD_DIR) if filename.endswith('.csv')]

    return csv_records

# Route to upload a CSV file
@app.post("/csv/")
async def upload_csv_file(file: UploadFile):
    # Ensure the uploads directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # csv_records.append({"name": file.filename})

    create_embeddings(file_path, '')

    return JSONResponse(content={"message": "CSV file uploaded successfully"}, status_code=201)


#####################################################################
# Search API #

# Create query object class
class Query(BaseModel):
    query_text: str


@app.post("/exhaustive_search/")
async def exhaustive_search(query: Query):

    # Get all csv files:
    base_path = 'semantic_matching/data/'
    csv_files = [base_path+file for file in os.listdir(base_path) if file.endswith('.csv')]

    k=20
    # Calculate top-k similarities
    top_k_results = batch_semantic_matching(query.query_text, csv_files, k).drop('Embeddings', axis=1)

    return Response(top_k_results.to_json(orient="records"), media_type="application/json")


@app.post("/ann_search/")
async def ann_search(query: Query):

    print("ANN SEARCH")
    client = QdrantClient("localhost", port=6333)

    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=model.encode(query.query_text).tolist(),
        limit=QDRANT_TOP_K,
    )

    data = []

    for hit in hits:
        payload_data = {
            'TableName': hit.payload.get('TableName', None),
            'CellValue': hit.payload.get('CellValue', None),
            'CellValue_Column': hit.payload.get('CellValue_Column', None),
            'SimilaritiyScores': hit.score
        }
        data.append(payload_data)

    df = pd.DataFrame(data)
    
    return Response(df.to_json(orient="records"), media_type="application/json")






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
    

