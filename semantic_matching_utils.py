import os
from os.path import isfile, join
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.spatial import distance
import umap
import hdbscan
import umap.plot
from umap.umap_ import nearest_neighbors
import joblib

# Helper Functions

#----------------------------------------------------------------------------------------------

def count_embeddings_per_cluster(cluster_labels):
    """
    Count the number of embeddings within each cluster and return the counts in a Pandas DataFrame.

    Parameters:
    cluster_labels (numpy.ndarray): Cluster labels assigned to each data point.

    Returns:
    pandas.DataFrame: A DataFrame with cluster labels and the count of embeddings in each cluster.
    """
    cluster_counts = pd.Series(cluster_labels).value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    cluster_counts = cluster_counts.sort_values(by='Cluster', ascending=True).reset_index(drop=True)

    return cluster_counts

#----------------------------------------------------------------------------------------------

def apply_elbow_method(input_embedding, max_n_of_clusters):
  """Apply the elbow method for K-Means algorithm to determine the optimal number of clusters
  """
  wcss = []
  for k in range(1, max_n_of_clusters):  # Try different values of k

      kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
      kmeans.fit(input_embedding)
      wcss.append(kmeans.inertia_)  # Inertia is the within-cluster sum of squares

  # Plot the Elbow Method graph
  plt.figure(figsize=(8, 5))
  plt.plot(range(1, max_n_of_clusters), wcss, marker='o', linestyle='-')
  plt.title('Elbow Method')
  plt.xlabel('Number of Clusters (k)')
  plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
  plt.xticks(range(1, max_n_of_clusters))
  plt.grid(True)

  # Find the optimal number of clusters (the "elbow" point)
  plt.annotate('Optimal k', xy=(3, wcss[2]), xytext=(4, wcss[4] + 300),
              arrowprops=dict(arrowstyle='->', lw=1.5),
              fontsize=12)
  plt.show()

#----------------------------------------------------------------------------------------------

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


def generate_umap_embeddings(n_neighbors, n_components, message_embeddings, pre_computed_knn=False, n_jobs = 50):
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

#----------------------------------------------------------------------------------------------
# K-MEANS clustering

def kmeans_clustering(text_embeddings, num_clusters):
    """
    Perform K-Means clustering on text embeddings and create a clustering index DataFrame with centroids only.

    Parameters:
    text_embeddings (numpy.ndarray or pandas DataFrame): Text embeddings for clustering.
    num_clusters (int): The number of clusters to create.

    Returns:
    tuple: A tuple containing cluster labels, clustering index (DataFrame), cluster centers, and runtime.
           - cluster_labels (numpy.ndarray): Cluster labels assigned to each data point.
           - clustering_index (pandas.DataFrame): A DataFrame containing cluster information with centroids.
           - cluster_centers (numpy.ndarray): Cluster centers.
           - runtime (float): The runtime of the K-Means clustering in seconds.
    """
    start_time = time.time()
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
    cluster_labels = kmeans.fit_predict(text_embeddings)
    cluster_centers = kmeans.cluster_centers_
    end_time = time.time()
    runtime = end_time - start_time

    # Create a clustering index DataFrame with centroids only
    clustering_index = pd.DataFrame({'Cluster': range(num_clusters)})
    clustering_index['Centroid'] = cluster_centers.tolist()

    print(f"K-Means finished. Runtime: {round(runtime, 2)}s")

    return cluster_labels, clustering_index, cluster_centers, runtime

#----------------------------------------------------------------------------------------------
# DBSCAN with Centroids

def dbscan_clustering(text_embeddings, eps, min_samples):

    start_time = time.time()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
    cluster_labels = dbscan.fit_predict(text_embeddings)
    end_time = time.time()
    runtime = end_time - start_time

    print(f"DBSCAN finished. Runtime: {round(runtime, 2)}s")

    # Create a DataFrame with the data and cluster labels
    df = pd.DataFrame(text_embeddings, columns=[f"Feature_{i}" for i in range(text_embeddings.shape[1])])
    df["Cluster"] = cluster_labels

    # Calculate cluster centroids
    cluster_centroids = df.groupby("Cluster").mean()

    # Create a clustering index DataFrame with centroids
    clustering_index = pd.DataFrame({'Cluster': cluster_centroids.index})
    clustering_index['Centroid'] = cluster_centroids.values.tolist()

    return cluster_labels, clustering_index, cluster_centroids.values, runtime


#----------------------------------------------------------------------------------------------
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

#----------------------------------------------------------------------------------------------
