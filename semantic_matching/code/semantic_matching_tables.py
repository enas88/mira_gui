import os
import time
import tqdm
import uuid
import umap
import torch
import socket
import joblib
import hdbscan
import numpy as np
import pandas as pd
from tqdm import tqdm 
from qdrant_client import models
from scipy.spatial import distance
from qdrant_client import QdrantClient
from umap.umap_ import nearest_neighbors
from semantic_matching.code.config import *
from sentence_transformers import SentenceTransformer, util
from qdrant_client.http.models import Distance, VectorParams


# Load SBERT model
model = SentenceTransformer(SBERT_PATH)

def create_embeddings(table_name, data_path, embeddings_path):
    """
    Creates embeddings for the target table as pt files

    Parameters: 
    - base_path : String: The base of the path. 
    - table_name: String: The name of the table to process. Will be appended to base_path

    Returns: String : A success message

    """ 
    
    if os.path.exists(data_path + table_name.replace('.csv', '.pt') ):
        return 'Embeddings file already exist'

    df = pd.read_csv(data_path + table_name)

    col_names = df.columns.tolist()
    df_columns = pd.DataFrame([col_names], columns=col_names)

    df = pd.concat([df_columns,df], ignore_index=True)
    df = df.map(lambda cell_value: str(cell_value))
    cell_values_flat = df.values.flatten()

    # Create embeddings for all cell values
    embeddings_list_flat = model.encode(cell_values_flat)

    # Save embeddings to pt file
    torch.save(embeddings_list_flat, embeddings_path + table_name.replace('.csv', '.pt') )

    return 'Embeddings created successfully'

#############################################################################################


def create_data(table_name, data_path, embeddings_path):
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
    print(f'Processing table: {table_name}')

    # Read csv data
    # base_path = 'semantic_matching/data/'
    df = pd.read_csv(data_path+'/'+table_name)

    # Add column names as first row
    col_names = df.columns.tolist()
    df_columns = pd.DataFrame([col_names], columns=col_names)
    df = pd.concat([df_columns,df], ignore_index=True)
    df = df.map(lambda cell_value: str(cell_value))
    cell_values_flat = df.values.flatten()

    # Create embeddings for all cell values
    # embeddings_list_flat = model.encode(cell_values_flat)

    # Load embeddings from a precomputed file
    embeddings_path = embeddings_path + '/' +table_name.replace('.csv', '.pt')
    assert os.path.exists(embeddings_path), "Embeddings path doesn't exist"
    embeddings_list_flat = torch.load(embeddings_path)
    
    # cell_numbers_flat = list(range(len(cell_values_flat)))

    # Duplicate column names for each cell value
    col_names_flat = col_names * int((len(cell_values_flat)/len(col_names)))

    # Final DataFrame
    name_and_embs = pd.DataFrame(list(zip([table_name.split('/')[-1]]*len(embeddings_list_flat), cell_values_flat , col_names_flat, embeddings_list_flat)), columns=['TableName','CellValue','CellValue_Column','Embeddings'])

    return name_and_embs

#############################################################################################


def get_top_k(query, input_df, k):
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
    query_emb = model.encode(query)
    cos_sims = util.cos_sim(query_emb, input_df['Embeddings']).numpy()[0]
    input_df['SimilaritiyScores'] = cos_sims
    input_df =  input_df.drop_duplicates(subset='CellValue').reset_index(drop=True)
    input_df =  input_df.sort_values('SimilaritiyScores', ascending=False)

    return input_df.iloc[:k]

#############################################################################################


def batch_semantic_matching(query, table_names, k):
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
    
    total_results_df = pd.DataFrame([], columns=['TableName','CellValue','CellValue_Column','Embeddings'])

    for table_name in table_names:

        # Calculate cosine similarities 
        res_df = create_data(table_name, BASE_DIR+DATA_DIR, BASE_DIR+EMBEDDINGS_DIR)
        total_results_df = pd.concat([total_results_df, res_df], ignore_index=True)

    # Get top-k similar results
    top_k_df = get_top_k(query, total_results_df, k).reset_index(drop=True)

    return top_k_df

##############################```` Qdrant````#################################################
# Functions for ANN and Qdrant


def create_collection(client , collection_name):
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
    print ('Creating collection:', collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

##############################```` Qdrant````#################################################

def delete_collection(client , collection_name):
    """
    Delete a collection with the specified name using Qdrant.

    Parameters:
    - client (QdrantClient): Quadrant client for communication.
    - collection_name (str): The name of the collection to be deleted.

    Returns:
    None

    """
    print('Deleting collection:', collection_name)
    client.delete_collection(collection_name)

##############################```` Qdrant````#################################################


def add_to_collection(table_name, table_path, embeddings_path, client , collection_name):
    """
    Adds a file into the specified collection. Works for .csv files for now. 
    We assume the embeddings for this file are already created in a separate pt file with the
    same name.

    Parameters:
    - table_name : String : The name of the csv file
    - table_name : String : The path for the folder the csv is located 
    - embeddings_path : String: The folder that contains the embeddings
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
    assert os.path.exists(table_path + table_name), "Table path does not exist"
    df = pd.read_csv(table_path + table_name)

    # Add column names as first row
    col_names = df.columns.tolist()
    df_columns = pd.DataFrame([col_names], columns=col_names)
    df = pd.concat([df_columns,df], ignore_index=True)
    cell_values_flat = df.values.flatten()

    # Load embeddings
    embeddings_path = embeddings_path + table_name.replace('.csv', '.pt')
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

def qdrant_is_working(host, port):
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

# Functions for the Efficient Search 

def process_csv(csv_file, embeddings_path):
    print(f'Processing table: {csv_file}')

    # Read CSV data
    df = pd.read_csv(csv_file)

    # Load precomputed embeddings from file
    embeddings_list = torch.load(embeddings_path)

    # Add column names as the first row
    col_names = df.columns.tolist()
    df_columns = pd.DataFrame([col_names], columns=col_names)
    df = pd.concat([df_columns, df], ignore_index=True)
    df = df.map(lambda cell_value: str(cell_value))
    cell_values_flat = df.values.flatten()

    # Final DataFrame
    name_and_embs = pd.DataFrame(
        list(zip([csv_file.split('/')[-1]] * len(embeddings_list), cell_values_flat, col_names * len(embeddings_list), embeddings_list)),
        columns=['TableName', 'CellValue', 'CellValue_Column', 'Embeddings'])

    return name_and_embs


def create_data_and_save_embeddings(csv_directory, embeddings_directory):
    all_data_frames = []
    # all_embeddings = []

    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

    for csv_file in tqdm(csv_files, desc='Processing CSV files'):
        csv_path = os.path.join(csv_directory, csv_file)
        embeddings_path = os.path.join(embeddings_directory, f'{csv_file.split(".")[0]}.pt')

        df = process_csv(csv_path, embeddings_path)
        all_data_frames.append(df)
        # all_embeddings.extend(embeddings)

    # Concatenate data frames
    merged_df = pd.concat(all_data_frames, ignore_index=True)

    # Save the merged data frame to a CSV file (optional)
    # merged_df.to_csv('merged_data.csv', index=False)
    all_embeddings = merged_df['Embeddings']
    # Save the embeddings array to a single file
    torch.save(merged_df['Embeddings'].tolist(), embeddings_directory+'/all_embeddings.pt')

    return merged_df, all_embeddings

# UMAP

def precompute_umap_knn(embeddings_array, n_neighbors, metric, save=False, filename="precomputed_knns.joblib"):
    """Calculates the k-NN's required for UMAP in order to speed up UMAP's algorithm
    """
    start_time = time.time()
    knn = nearest_neighbors(
                        embeddings_array,
                        n_neighbors=n_neighbors,
                        metric=metric,
                        metric_kwds=None,
                        angular=False,
                        random_state=None
                        )
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Computing k-NNs process finished. Runtime: {round(runtime, 2)}s")

    if save:
        joblib.dump(knn, filename)

    return knn


def generate_umap_embeddings(n_neighbors, n_components, message_embeddings, pre_computed_knn=False, n_jobs = -1):
    """
    Generate UMAP embeddings for a given set of message embeddings.

    Parameters:
    n_neighbors (int): The number of neighbors to consider during UMAP dimensionality reduction.
    n_components (int): The number of components in the reduced space.
    message_embeddings (numpy.ndarray): The message embeddings for which UMAP embeddings will be generated.

    Returns:
    numpy.ndarray: UMAP embeddings in the reduced space.
    float: Running time in seconds.
    """
    
    # Create a UMAP instance with specified parameters
    if pre_computed_knn:
        umap_model = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, precomputed_knn = pre_computed_knn, n_jobs=n_jobs, metric='cosine')
    else:
        umap_model = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components,n_jobs=n_jobs ,metric='cosine')

    start_time = time.time()

    # Fit and transform the message embeddings to the reduced space
    umap_trans = umap_model.fit(message_embeddings)
    umap_embeddings = umap_trans.transform(message_embeddings)

    end_time = time.time()
    runtime = end_time - start_time
    print(f"UMAP finished. Runtime: {round(runtime, 2)}s")

    return umap_embeddings, umap_trans, runtime


# HDBSCAN with Medoids

def hdbscan_clustering(text_embeddings, min_samples):

    start_time = time.time()
    clusterer = hdbscan.HDBSCAN(min_samples=min_samples, metric='euclidean', gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(text_embeddings)
    end_time = time.time()
    runtime = end_time - start_time

    print(f"HDBSCAN finished. Runtime: {round(runtime, 2)}s")

    # Create a DataFrame with the data and cluster labels
    df = pd.DataFrame(text_embeddings, columns=[f"Feature_{i}" for i in range(text_embeddings.shape[1])])
    df["Cluster"] = cluster_labels

    # Calculate cluster medoids
    unique_clusters = np.unique(cluster_labels)
    cluster_medoids = []

    for cluster_id in unique_clusters:
        cluster_points = df[df["Cluster"] == cluster_id].iloc[:, :-1].values  # Get cluster points

        # Calculate pairwise distances within the cluster
        pairwise_distances = distance.cdist(cluster_points, cluster_points, 'euclidean')

        # Sum the distances for each point to get the total distance
        total_distances = np.sum(pairwise_distances, axis=1)

        # Find the index of the point with the smallest total distance (medoid)
        medoid_index = np.argmin(total_distances)

        # Get the medoid point
        medoid = cluster_points[medoid_index]

        cluster_medoids.append(medoid)

    # Create a clustering index DataFrame with medoids
    clustering_index = pd.DataFrame({'Cluster': unique_clusters})
    clustering_index['Medoid'] = cluster_medoids

    return cluster_labels, clustering_index, cluster_medoids, runtime



def add_to_collection_clustered(table_name, client, embeddings_path, collection_name):
    """
    Adds a file into the specified collection. Works for .csv files for now. 
    We assume the embeddings for this file are already created in a separate pt file with the
    same name.

    Parameters:
    - table_name : String : The path of the csv file
    - client : Qdrant client object
    - embeddings_path: String: The path to the embeddings pt file
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
    
    # Prepare table
    print(f"QDRANT: Adding table {table_name} to {collection_name} collection")
    assert os.path.exists(table_name), "Table path does not exist"
    df = pd.read_csv(table_name)

    # Load embeddings
    assert os.path.exists(embeddings_path), "Embeddings path doesn't exist"
    embeddings_list_flat = torch.load(embeddings_path)

    df['Embeddings'] = embeddings_list_flat

    df_unduplicated = df.drop_duplicates(subset='CellValue').reset_index(drop=True)

    df_records = df_unduplicated.to_dict('records')


    # Add to collection

    client.upload_records(
    collection_name=COLLECTION_NAME_CLUSTERED,
    records=[
        models.Record(
            id=str(uuid.uuid4()), vector=doc['Embeddings'].tolist(), payload={key: value for key, value in doc.items() if key!='Embeddings' }
        )
        for idx, doc in enumerate(df_records)
    ],
    )


    return "Success"


def cluster_search(query_text, top_k_results, top_k_clusters,  clustering_index_path, umap_trans, client, collection_name):
    
  clustering_index = joblib.load(clustering_index_path)

  query_text_emb = model.encode(query_text)

  query_text_emb_umap = umap_trans.transform([query_text_emb])

  similarities = util.cos_sim(query_text_emb_umap[0], clustering_index['Medoid'].values.tolist())

  clustering_index['Similarity'] = similarities.numpy()[0]
  clustering_index = clustering_index.sort_values('Similarity', ascending=False)

  top_k_clusters = clustering_index.iloc[:top_k_clusters]
  top_k_clusters = top_k_clusters['Cluster'].values.tolist()

  print("EFFICIENT SEARCH")
  client = QdrantClient("localhost", port=6333)

  hits = client.search(
      collection_name=collection_name,
      query_vector=query_text_emb.tolist(),
      query_filter=models.Filter(
      must=[models.FieldCondition(key="Cluster", match=models.MatchAny(any=top_k_clusters))]
      ),
      limit=top_k_results,
  )

  data = []

  for hit in hits:
      payload_data = {
          'TableName': hit.payload.get('TableName', None),
          'CellValue': hit.payload.get('CellValue', None),
          'CellValue_Column': hit.payload.get('CellValue_Column', None),
          'Cluster': hit.payload.get('Cluster', None),
          'SimilaritiyScores': hit.score
      }
      data.append(payload_data)

  df_efficient = pd.DataFrame(data)

  return df_efficient