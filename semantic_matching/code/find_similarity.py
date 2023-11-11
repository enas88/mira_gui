import os
from os.path import isfile, join
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from prettytable import PrettyTable

def readFiles():

	datasetpath =  f"../data/GroundTruth_Data/"
	embeddingspath = f"../data/newdataembedding/"

	files = [f for f in os.listdir(datasetpath) if (isfile(join(datasetpath, f)) and f.endswith('.csv'))]
	embeddingsFiles = [f for f in os.listdir(embeddingspath) if (isfile(join(embeddingspath, f)) and f.endswith('.pt'))]
	
	embeddings = torch.empty(0, device = 'cpu')
	for embeddingsFile in embeddingsFiles:
		embeddingsOffile = torch.load(embeddingspath+ embeddingsFile, map_location=torch.device('cpu'))
		embeddings = torch.concat([embeddings, embeddingsOffile])

	
	filesContent = {}
	content = []
	
	for file in files:
		try:
			df = pd.read_csv( datasetpath + file, encoding = "ISO-8859-1", on_bad_lines = "skip")
			filesContent[file] = df.columns.tolist()
			content += df.columns.tolist()
			for row in df.values:
				filesContent[file] += list(map(str,row))
				content += list(map(str,row))
		except Exception as e:
			print(f"Error in reading a csv file: file name: {file}, Error message: {e}")
		
	return filesContent, content, embeddings

def getMeXResults(X,top_results, filesContent, content):
	XResults = []
	i = 0
	cellsFileCombi = []
	for score, idx in zip(top_results[0], top_results[1]):
		if float("{:.4f}".format(score)) >= 0.3:
			for filename, val in filesContent.items():
				if content[idx] in val:
					if (filename, content[idx]) not in cellsFileCombi:
						cellsFileCombi.append((filename, content[idx]))
						XResults.append((filename, content[idx], score))
						i += 1
						if i == X:
							return XResults
	return XResults

def results(query, filesContent, content, embeddings):   
	
	symmetric_embedder = SentenceTransformer('all-mpnet-base-v2')
	results = {}
	
	query_embedding = symmetric_embedder.encode(query, convert_to_tensor = True)
	top_k = min(10, len(embeddings))
	cos_scores = util.cos_sim(query_embedding, embeddings)[0]
	top_results = torch.topk(cos_scores, len(embeddings), sorted = True)
	
	results = getMeXResults(top_k, top_results, filesContent, content)
	return results

def matching():
	filesContent, content, embeddings = readFiles()
	query = "Computer Scientist"
	result = results(query, filesContent, content, embeddings)
	print("============================")
	print(f"your query is: {query}")
	
	table = [['Filename', 'Cell Value', 'Score']]
	for i in result:
		table.append([i[0], i[1], "{:.4f}".format(i[2])])
	tab = PrettyTable(table[0])
	tab.add_rows(table[1:])
	print(tab)
		
if __name__ == "__main__":
	matching()
