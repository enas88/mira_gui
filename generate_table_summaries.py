import os
import pandas as pd
import json
from collections import Counter
from pathlib import Path

# Define the paths
DATA_DIR = "semantic_matching/data/"
UPLOAD_DIR = "uploads/"

# Output file for table summaries
OUTPUT_FILE = "table_summaries.json"

# Helper function to extract keywords
def extract_top_keywords(df, top_n=3):
    # Concatenate all text content from the dataframe
    text_content = " ".join(df.astype(str).values.flatten())
    words = text_content.split()
    # Use Counter to get most common words (keywords)
    common_words = Counter(words).most_common(top_n)
    return [word for word, _ in common_words]

# Function to process a single CSV file
def process_csv(file_path, dataset_type="generic"):
    df = pd.read_csv(file_path)
    table_info = {
        "table_name": Path(file_path).stem,
        "rows": df.shape[0],
        "columns": df.shape[1],
        "top_keywords": extract_top_keywords(df),
        "dataset_type": dataset_type
    }
    return table_info

# Main function to gather summaries for all CSV files
def generate_table_summaries():
    summaries = []

    # Process files in DATA_DIR
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".csv"):
            file_path = os.path.join(DATA_DIR, filename)
            summaries.append(process_csv(file_path, dataset_type="original"))

    # Process files in UPLOAD_DIR
    for filename in os.listdir(UPLOAD_DIR):
        if filename.endswith(".csv"):
            file_path = os.path.join(UPLOAD_DIR, filename)
            summaries.append(process_csv(file_path, dataset_type="uploaded"))

    # Save summaries to a JSON file
    with open(OUTPUT_FILE, "w") as f:
        json.dump(summaries, f, indent=4)

    print(f"Table summaries saved to {OUTPUT_FILE}")

# Run the function to generate summaries
if __name__ == "__main__":
    generate_table_summaries()
