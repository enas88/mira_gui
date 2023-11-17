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
    """
    Creates embeddings for the target table as pt files

    Parameters: 
    - base_path : String: The base of the path. 
    - table_name: String: The name of the table to process. Will be appended to base_path

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
    """
    Process a CSV table, generate embeddings for its cell values, and create a DataFrame.

    Parameters:
     - table_name (str): The path to the CSV file containing the table data.

    Returns:
    pandas.DataFrame: A DataFrame containing information about each cell, including
    the table name, cell value, corresponding column name, and embeddings.

    Notes:
    - The CSV file is expected to have headers, and the first row is considered as
      the header row.
    - Embeddings for cell values are loaded from a precomputed file with a '.pt' extension,
      and the file should be located in the same directory as the CSV file.

    """

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

    # Load embeddings from a precomputed file
    embeddings_path = table_name.replace('.csv', '.pt')
    assert os.path.exists(embeddings_path), "Embeddings path doesn't exist"
    embeddings_list_flat = torch.load(embeddings_path)
    
    # cell_numbers_flat = list(range(len(cell_values_flat)))

    # Duplicate column names for each cell value
    col_names_flat = col_names * int((len(cell_values_flat)/len(col_names)))

    # Final DataFrame
    name_and_embs = pd.DataFrame(list(zip([table_name.split('/')[-1]]*len(embeddings_list_flat), cell_values_flat , col_names_flat, embeddings_list_flat)), columns=['TableName','CellValue','CellValue_Column','Embeddings'])

    return name_and_embs

#############################################################################################
    """
    Retrieve the top k similar cell values from the input DataFrame based on a query.

    Parameters:
    - query (str): The query string for which to find similar cell values.
    - input_df (pandas.DataFrame): The DataFrame containing cell information,
      including cell values, column names, embeddings, and similarity scores.
    - k (int): The number of top similar cell values to retrieve.

    Returns:
    pandas.DataFrame: A DataFrame containing the top k similar cell values,
    sorted by similarity scores in descending order.

    Notes:
    - The input DataFrame is expected to have a 'Embeddings' column containing
      precomputed embeddings for each cell value.
    - Similarity scores are computed based on cosine similarity between the query
      embedding and the embeddings in the DataFrame.
    - The returned DataFrame includes columns such as 'CellValue', 'CellValue_Column',
      'Embeddings', and 'SimilarityScores'.
    - Duplicate cell values are removed before computing similarity scores.

    """

def get_top_k(query, input_df, k):

    query_emb = model.encode(query)
    cos_sims = util.cos_sim(query_emb, input_df['Embeddings']).numpy()[0]
    input_df['SimilaritiyScores'] = cos_sims
    input_df =  input_df.drop_duplicates(subset='CellValue').reset_index(drop=True)
    input_df =  input_df.sort_values('SimilaritiyScores', ascending=False)

    return input_df.iloc[:k]

#############################################################################################
    """
    Perform batch semantic matching across multiple tables and retrieve top-k similar results.

    Parameters:
    - query (str): The query string for which to find similar cell values.
    - table_names (list of str): A list of paths to CSV files containing table data.
    - k (int): The number of top similar cell values to retrieve.

    Returns:
    pandas.DataFrame: A DataFrame containing the top k similar cell values
    from all tables, sorted by similarity scores in descending order.

    Notes:
    - Each table is processed using the create_data function to generate embeddings.
    - The resulting DataFrames are concatenated to form a total results DataFrame.
    - Similarity scores are computed based on cosine similarity between the query
      embedding and the embeddings in the total results DataFrame.
    - The returned DataFrame includes columns such as 'TableName', 'CellValue',
      'CellValue_Column', 'Embeddings', and 'SimilarityScores'.
    - Duplicate cell values are removed before computing similarity scores.

    """

def batch_semantic_matching(query, table_names, k):

    total_results_df = pd.DataFrame([], columns=['TableName','CellValue','CellValue_Column','Embeddings'])

    for table_name in table_names:

        # Calculate cosine similarities 
        res_df = create_data(table_name)
        total_results_df = pd.concat([total_results_df, res_df], ignore_index=True)

    # Get top-k similar results
    top_k_df = get_top_k(query, total_results_df, k).reset_index(drop=True)

    return top_k_df

##############################```` Qdrant````#################################################
# Functions for ANN and Qdrant

    """
    Create a collection with the specified name using Qdrant.

    Parameters:
    - client (QdrantClient): Quadrant client for communication.
    - collection_name (str): The name of the collection to be created.

    Returns:
    None

    Notes:
    - The function communicates with Qdrant to create a collection with the given name.
    - Make sure the Qdrant client is properly initialized before calling this function.
    - Adjust the `QdrantClient` type based on the actual client used in your code.
    - Provide any additional parameters required by the Qdrant client for collection creation.

    """
def create_collection(client , collection_name):

    print ('Creating collection:', collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

##############################```` Qdrant````#################################################
    """
    Delete a collection with the specified name using Qdrant.

    Parameters:
    - client (QdrantClient): Quadrant client for communication.
    - collection_name (str): The name of the collection to be deleted.

    Returns:
    None

    """
def delete_collection(client , collection_name):

    print('Deleting collection:', collection_name)
    client.delete_collection(collection_name)

##############################```` Qdrant````#################################################


def add_to_collection(table_name, client , collection_name):
    """
    Adds a file into the specified collection. Works for .csv files for now. 
    We assume the embeddings for this file are already created in a separate pt file with the
    same name.

    Parameters:
    - table_name : String : The path of the csv file
    - client : Qdrant client object
    - collection_name : String: The collection name to use

    Returns :
    - String : A success message

    Notes:
    - The function assumes that the embeddings for the CSV file are already created
      in a separate '.pt' file with the same name.
    - The CSV file is expected to have headers, and the first row is considered as
      the header row.
    - Duplicate cell values are removed before adding records to the collection.
    - Records are uploaded to the Quadrant collection, including vectors and metadata.
    - Adjust the `QuadrantClient` type based on the actual client used in your code.
    - Provide the correct field names and conditions based on your vector schema.

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
#############################################################################################
    """
    Check if Qdrant is working by attempting to establish a connection to the specified host and port.

    Parameters:
    - host (str): The host address where Qdrant is running.
    - port (int): The port number on which Qdrant is listening.

    Returns:
    bool: True if a connection can be established, indicating Qdrant is working; False otherwise.

    Notes:
    - The function uses a socket connection to check if Qdrant is working at the specified host and port.
    - Adjust the timeout value as needed based on the expected response time.
    - Returns True if a connection is successfully established (result code 0), indicating Qdrant is working.
    - Returns False if a connection cannot be established.

    """
def qdrant_is_working(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)  # Adjust the timeout as needed
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

#############################################################################################
def count_table_occurencies(table_name, client, collection_name):
    """
    Count the occurrences of a specific table in the specified collection.

    Parameters:
    - table_name (str): The name of the table to count occurrences for.
    - client (QdrantClient): Quadrant client for communication.
    - collection_name (str): The name of the collection containing the vector data.

    Returns:
    int: The number of occurrences of the specified table in the collection.

    Notes:
    - The function uses the Qdrant client to count occurrences based on the provided table name.
    - The count is specific to the specified collection.
    - Adjust the `QdrantClient` type based on the actual client used in your code.
    - Provide the correct field name and match condition based on your vector schema.

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

#############################################################################################


