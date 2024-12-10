import os
import time
import json 
import shutil
import logging

import numpy as np
import pandas as pd
import mysql.connector
from pathlib import Path
from mysql.connector import Error
from typing import Optional, Dict

from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi import FastAPI, File, UploadFile, Response, HTTPException, Request

import qdrant_client
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

from pydantic import BaseModel

# from semantic_matching.code.config import *

import sentence_transformers
from sentence_transformers import SentenceTransformer, util
from semantic_matching.code.semantic_matching_tables import *

logging.basicConfig(level=logging.INFO)


#####################################################################
# Initialization

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
client = QdrantClient("localhost", port=6333)


#####################################################################

@app.get("/")
async def get_form():
    return FileResponse("templates/dashboard.html")

@app.get("/dashboard")
async def get_dashboard():
    return FileResponse("templates/dashboard.html")

@app.get("/about")
async def get_about():
    return FileResponse("templates/about.html")

@app.get("/catalog")
async def get_form():
    return FileResponse("templates/catalog.html")

@app.get("/useful")
async def get_form():
    return FileResponse("templates/useful.html")

@app.get("/exhaustive")
async def get_form():
    return FileResponse("templates/index.html")

@app.get("/ann")
async def get_form():
    return FileResponse("templates/ann.html")

@app.get("/optimized")
async def get_form():
    return FileResponse("templates/optimized.html")

@app.get("/upload")
async def get_form():
    return FileResponse("templates/upload.html")

@app.get("/view")
async def get_form():
    return FileResponse("templates/view.html")

@app.get("/Registered_Data")
async def get_form():
    return FileResponse("templates/Registered_Data.html")

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

    # A list to store CSV records
    csv_records = []
    if len(os.listdir(UPLOAD_DIR))>0 :
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

    create_embeddings(file.filename, UPLOAD_DIR, BASE_DIR_EMBEDDINGS_DIR_SAVE )

    # client = QdrantClient(host="localhost", port=6333)

    if qdrant_is_working("localhost", 6333):
        add_to_collection(file.filename, UPLOAD_DIR, BASE_DIR_EMBEDDINGS_DIR_SAVE , client, COLLECTION_NAME)

    return JSONResponse(content={"message": "CSV file uploaded successfully"}, status_code=201)

#####################################################################
# Efficient search

@app.get("/run_umap")
async def run_umap():

    # Combine all of the available data and create embeddings
    merged_df, all_embeddings = create_data_and_save_embeddings(BASE_DIR_DATA_DIR , BASE_DIR_DATA_DIR )

    # Precompute knn's for UMAP
    all_embeddings_array = np.vstack(all_embeddings)
    knns_jobs = precompute_umap_knn(all_embeddings_array, 20, "euclidean", False, filename=BASE_DIR+JOBLIBS_DIR+"/allfiles_umap.joblib")

    # Apply UMAP
    umap_embeddings, umap_trans, _ = generate_umap_embeddings(20, 2, all_embeddings_array, pre_computed_knn=knns_jobs, n_jobs = -1)

    joblib.dump(umap_trans, BASE_DIR_DATA_DIR +"/umap_transformer.joblib")
    joblib.dump(umap_embeddings, BASE_DIR_DATA_DIR +"/umap_embeddings.joblib")

    return JSONResponse(content={"message": "Created embeddings and saved UMAP objects"}, status_code=201)



#####################################################################
# Search API #

# Create query object class
class Query(BaseModel):
    query_text: str

class Dataset(BaseModel):
    data: list[list]  # Assuming the input data is a list of lists
    columns: list[str]  # Column names for the DataFrame

@app.post("/usefulness_search/")
async def exhaustive_search(query: Query):

    # Get all csv files:
    csv_files = [file for file in os.listdir(UPLOAD_DIR) if file.endswith('.csv')]

    k=20
    # Calculate top-k similarities
    top_k_results = batch_semantic_matching(query.query_text, csv_files, k).drop('Embeddings', axis=1)

    return Response(top_k_results.to_json(orient="records"), media_type="application/json")

#####################################################################
# Add new record to the dataset catalog #


class Dataset(BaseModel):
    id: str
    name: str
    date: str
    type: str
    url: Optional[str] = None  # Make URL optional
    path: Optional[str] = None  # Make Path optional
    username: str
    password: str
    metadata: Dict[str, str]
    add_dataset: str = "no"  # New field with a default value of "no"

class Dataset(BaseModel):
    id: str
    name: str
    date: str
    type: str
    url: str
    path: str
    username: str
    password: str
    add_dataset: str
    metadata: dict

@app.post("/dataset_catalog/")
async def add_dataset(dataset: Dataset):
    try:
        # Load existing data
        with open("dataset_catalog.json", "r") as file:
            data = json.load(file)

        # Append new dataset
        data.append(dataset.dict())

        # Save updated data
        with open("dataset_catalog.json", "w") as file:
            json.dump(data, file, indent=4)

        return {"message": "Dataset added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
#####################################################################


@app.post("/exhaustive_search/")
async def exhaustive_search(query: Query):
    # Get all CSV files:
    csv_files = [file for file in os.listdir(UPLOAD_DIR) if file.endswith('.csv')]

    k = 20
    # Calculate top-k similarities
    top_k_results = batch_semantic_matching(query.query_text, csv_files, k).drop('Embeddings', axis=1)

    # Rename the score column to match what your script expects if necessary
    if 'score' in top_k_results.columns:
        top_k_results.rename(columns={'score': 'SimilarityScores'}, inplace=True)

    # Ensure all other required columns are present, even if they are empty
    required_columns = ['TableName', 'CellValue', 'CellValue_Column', 'SimilarityScores']
    for col in required_columns:
        if col not in top_k_results.columns:
            top_k_results[col] = None  # Add missing columns as empty

    return Response(top_k_results.to_json(orient="records"), media_type="application/json")


#####################################################################
@app.post("/ann_search/")
async def ann_search(query: Query):

    print("ANN SEARCH")
    
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
            'SimilarityScores': hit.score
        }
        data.append(payload_data)

    df = pd.DataFrame(data)
    
    return Response(df.to_json(orient="records"), media_type="application/json")


#####################################################################
@app.post("/efficient_search")
async def efficient_search(query: Query):
    # Load table summaries JSON
    table_summaries_path = 'semantic_matching/table_summaries.json'
    with open(table_summaries_path, 'r') as f:
        table_summaries = json.load(f)

    # Perform clustering and search
    top_k_results = 10
    top_k_clusters = 20
    clustering_index_path = 'semantic_matching/merged_data/clustering_index.joblib'
    umap_trans_path = 'semantic_matching/merged_data/umap_trans.joblib'

    umap_trans = joblib.load(umap_trans_path)
    
    df_efficient = cluster_search(
        query.query_text, 
        top_k_results, 
        top_k_clusters, 
        clustering_index_path, 
        umap_trans, 
        client, 
        COLLECTION_NAME_CLUSTERED
    )

    # Merge results with table summaries
    formatted_results = []
    for _, row in df_efficient.iterrows():
        table_name = row['TableName'].replace('.csv', '')
        table_info = table_summaries[table_name]
        
        formatted_results.append({
            "TableName": table_name,
            "Rows": table_info['rows'],
            "Columns": table_info['columns'],
            "Type": table_info['dataset_type'],
            "CellValue": row['CellValue'],
            "CellValue_Column": row['CellValue_Column'],
            "SimilarityScores": row['SimilaritiyScores']
        })   

    return Response(json.dumps(formatted_results), media_type="application/json")

@app.get("/get_table/{dataset_name}")
async def get_table(dataset_name: str):
    # Logic to load the dataset table (e.g., from a CSV or database)
    # Assuming the table is loaded as a DataFrame
    df = pd.read_csv(f"semantic_matching/data/{dataset_name}.csv")  # Adjust path as needed
    data = {
        "columns": df.columns.tolist(),
        "data": df.values.tolist()
    }
    return JSONResponse(content=data)

#####################################################################
@app.get("/download/{dataset_name}")
async def download_dataset(dataset_name: str):
    # Path to your dataset files
    file_path = f"semantic_matching/data/{dataset_name}.csv"  # Adjust path as needed

    # Check if the file exists
    if not os.path.isfile(file_path):
        return {"error": "File not found."}
    
    # Return the file for download
    return FileResponse(path=file_path, filename=f"{dataset_name}.csv", media_type="text/csv")

#####################################################################
@app.get("/api/registered_data")
async def get_registered_data():
    try:
        # Load the JSON file
        with open("dataset_catalog.json", "r") as file:
            data = json.load(file)
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    


#####################################################################
    # Path to your JSON file
JSON_FILE = "dataset_catalog.json"

@app.delete("/api/delete/{record_id}")
async def delete_record(record_id: str):
    try:
        if not os.path.exists(JSON_FILE):
            raise HTTPException(status_code=404, detail="JSON file not found.")
        
        # Load existing data
        with open(JSON_FILE, "r") as file:
            data = json.load(file)
        
        # Filter out the record to delete
        updated_data = [record for record in data if record["id"] != record_id]
        
        # Check if a record was deleted
        if len(updated_data) == len(data):
            raise HTTPException(status_code=404, detail="Record not found.")
        
        # Save the updated data back to the file
        with open(JSON_FILE, "w") as file:
            json.dump(updated_data, file, indent=4)
        
        return {"message": "Record deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    #####################################################################

    # Update record in the JSON file
@app.post("/api/process_record/{record_id}")
async def process_single_record(record_id: str):
    try:
        # Load dataset catalog
        with open("dataset_catalog.json", "r") as file:
            data = json.load(file)

        # Find the record by ID
        record = next((rec for rec in data if rec["id"] == record_id), None)
        if not record:
            raise HTTPException(status_code=404, detail="Record not found.")

        # Check if the record is already processed
        if record.get("processed"):
            raise HTTPException(status_code=400, detail="Record is already processed.")

        # Check if the record is eligible for processing
        if record["add_dataset"] != "yes":
            raise HTTPException(status_code=400, detail="Record is not eligible for processing.")

        # Validate file existence
        file_path = record["path"]
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=400, detail=f"File not found: {file_path}")

        # Extract the file name from the path
        file_name = os.path.basename(file_path)

        # Process the file (e.g., create embeddings, add to collection)
        create_embeddings(file_name, UPLOAD_DIR, BASE_DIR_EMBEDDINGS_DIR_SAVE)
        if qdrant_is_working("localhost", 6333):
            add_to_collection(file_name, UPLOAD_DIR, BASE_DIR_EMBEDDINGS_DIR_SAVE, client, COLLECTION_NAME)

        # Mark the record as processed
        record["processed"] = True

        # Save the updated JSON data back to the file
        with open("dataset_catalog.json", "w") as file:
            json.dump(data, file, indent=4)

        return {"success": True, "message": f"Record {record_id} processed successfully."}

    except FileNotFoundError as fnf_error:
        logging.error(f"File not found: {fnf_error}")
        raise HTTPException(status_code=404, detail=str(fnf_error))

    except HTTPException as http_error:
        logging.error(f"HTTP error: {http_error.detail}")
        raise http_error

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.post("/redirect-to-dashboard")
async def redirect_to_dashboard():
    # Redirect to the /dashboard route
    return RedirectResponse(url="/dashboard", status_code=303)

@app.post("/redirect-to-about")
async def redirect_to_about():
    # Redirect to the /about route
    return RedirectResponse(url="/about", status_code=303)