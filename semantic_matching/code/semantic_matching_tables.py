import os
import uuid
import torch
import socket
import numpy as np
import pandas as pd
from config import *
from qdrant_client import models
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, util
from qdrant_client.http.models import Distance, VectorParams

# Load SBERT model
model = SentenceTransformer(SBERT_PATH)

def create_embeddings(table_name, base_path):
    """Creates embeddings for the target table as pt files

        Parameters: 
                    base_path : String: The base of the path. 
                    table_name: String: The name of the table to process. Will be appended to base_path

        Returns: String : A success message

    """ 
    print(base_path + table_name)
    
    if os.path.exists(base_path + table_name.replace('.csv', '.pt') ):
        return 'Embeddings file already exist'

    df = pd.read_csv(base_path + table_name)

    col_names = df.columns.tolist()
    df_columns = pd.DataFrame([col_names], columns=col_names)

    df = pd.concat([df_columns,df], ignore_index=True)
    df = df.map(lambda cell_value: str(cell_value))
    cell_values_flat = df.values.flatten()

    # Create embeddings for all cell values
    embeddings_list_flat = model.encode(cell_values_flat)

    # Save embeddings to pt file
    torch.save(embeddings_list_flat, base_path + table_name.replace('.csv', '.pt') )

    return 'Embeddings created successfully'

#############################################################################################

def create_data(table_name):

    print(f'Processing table: {table_name}')

    # Read csv data
    # base_path = 'semantic_matching/data/'
    df = pd.read_csv(table_name)

    # Add column names as first row
    col_names = df.columns.tolist()
    df_columns = pd.DataFrame([col_names], columns=col_names)
    df = pd.concat([df_columns,df], ignore_index=True)
    df = df.map(lambda cell_value: str(cell_value))
    cell_values_flat = df.values.flatten()

    # Create embeddings for all cell values
    # embeddings_list_flat = model.encode(cell_values_flat)
    embeddings_path = table_name.replace('.csv', '.pt')
    assert os.path.exists(embeddings_path), "Embeddings path doesn't exist"
    embeddings_list_flat = torch.load(embeddings_path)
    
    # cell_numbers_flat = list(range(len(cell_values_flat)))

    col_names_flat = col_names * int((len(cell_values_flat)/len(col_names)))

    # Final DataFrame
    name_and_embs = pd.DataFrame(list(zip([table_name.split('/')[-1]]*len(embeddings_list_flat), cell_values_flat , col_names_flat, embeddings_list_flat)), columns=['TableName','CellValue','CellValue_Column','Embeddings'])

    return name_and_embs

#############################################################################################

def get_top_k(query, input_df, k):

    query_emb = model.encode(query)
    cos_sims = util.cos_sim(query_emb, input_df['Embeddings']).numpy()[0]
    input_df['SimilaritiyScores'] = cos_sims
    input_df =  input_df.drop_duplicates(subset='CellValue').reset_index(drop=True)
    input_df =  input_df.sort_values('SimilaritiyScores', ascending=False)

    return input_df.iloc[:k]

#############################################################################################

def batch_semantic_matching(query, table_names, k):

    total_results_df = pd.DataFrame([], columns=['TableName','CellValue','CellValue_Column','Embeddings'])

    for table_name in table_names:

        # Calculate cosine similarities 
        res_df = create_data(table_name)
        total_results_df = pd.concat([total_results_df, res_df], ignore_index=True)

    # Get top-k similar results
    top_k_df = get_top_k(query, total_results_df, k).reset_index(drop=True)

    return top_k_df

#############################################################################################
# Functions for ANN and Qdrant


def create_collection(client , collection_name):

    print ('Creating collection:', collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )


def delete_collection(client , collection_name):

    print('Deleting collection:', collection_name)
    client.delete_collection(collection_name)


def add_to_collection(table_name, client , collection_name):
    """Adds a file into the specified collection. Works for .csv files for now. 
       We assume the embeddings for this file are already created in a separate pt file with the
       same name.

        Parameters: table_name : String : The path of the csv file
                    client : Qdrant client object
                    collection_name : String: The collection name to use

        Returns : String : A success message
    """
    ########################################################################
    # Prepare table
    print(f"QDRANT: Adding table {table_name} to {collection_name} collection")
    assert os.path.exists(table_name), "Table path does not exist"
    df = pd.read_csv(table_name)

    # Add column names as first row
    col_names = df.columns.tolist()
    df_columns = pd.DataFrame([col_names], columns=col_names)
    df = pd.concat([df_columns,df], ignore_index=True)
    cell_values_flat = df.values.flatten()

    # Load embeddings
    embeddings_path = table_name.replace('.csv', '.pt')
    assert os.path.exists(embeddings_path), "Embeddings path doesn't exist"
    embeddings_list_flat = torch.load(embeddings_path)

    col_names_flat = col_names * int((len(cell_values_flat)/len(col_names)))

    # Final DataFrame
    name_and_embs = pd.DataFrame(list(zip([table_name.split('/')[-1]]*len(embeddings_list_flat), cell_values_flat , col_names_flat, embeddings_list_flat)), columns=['TableName','CellValue','CellValue_Column','Embeddings'])
    name_and_embs =  name_and_embs.drop_duplicates(subset='CellValue').reset_index(drop=True)
    name_and_embs_dict = name_and_embs.to_dict('records')

    ########################################################################
    # Add to collection

    client.upload_records(
    collection_name=collection_name,
    records=[
        models.Record(
            id=str(uuid.uuid4()), vector=doc["Embeddings"].tolist(), payload={key: value for key, value in doc.items() if key != 'Embeddings'}
        )
        for idx, doc in enumerate(name_and_embs_dict)
    ],
    )

    return "Success"


def count_table_occurencies(table_name, client, collection_name):
    """ Counts how many occurencies there are in the specified collection for the specified table
    """
    print(f"Counting occurencies of table: {table_name}")

    result = client.count(
    collection_name=collection_name,
    count_filter=models.Filter(
        must=[
             models.FieldCondition(key="TableName", match=models.MatchValue(value=table_name)),
        ]
    ),
    exact=True,
    )

    return result.count


def qdrant_is_working(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)  # Adjust the timeout as needed
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0


