import os
import time
import json 
import shutil
import logging

import numpy as np
import pandas as pd
from pathlib import Path

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

@app.post("/dataset_catalog/")
async def read_dataset(dataset: Dataset):
    # Define the path to the CSV file
    file_path = Path("Catalogs/Datasets_Catalog.csv")
    
    # Check if the file exists
    if file_path.exists() and file_path.is_file():
        # If it exists, append without headers
        mode = 'a'
        header = False
    else:
        # If not, create a new file with headers 
        mode = 'w'
        header = True

    # Convert the data to a DataFrame
    df = pd.DataFrame(dataset.data, columns=dataset.columns)

    # Try to append or write to the CSV file
    try:
        df.to_csv(file_path, mode=mode, index=False, header=header)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while saving the dataset: {str(e)}")
    
    # Respond with success message
    return {"message": "Dataset received and appended to DataFrame"}


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
    print(formatted_results)    
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


@app.get("/download/{dataset_name}")
async def download_dataset(dataset_name: str):
    # Path to your dataset files
    file_path = f"semantic_matching/data/{dataset_name}.csv"  # Adjust path as needed

    # Check if the file exists
    if not os.path.isfile(file_path):
        return {"error": "File not found."}
    
    # Return the file for download
    return FileResponse(path=file_path, filename=f"{dataset_name}.csv", media_type="text/csv")