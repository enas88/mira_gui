import os
import pandas as pd
import json
from collections import Counter
from pathlib import Path

# Define paths
DATA_DIR = "semantic_matching/data/"
UPLOAD_DIR = "uploads/"
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
    table_name = Path(file_path).stem
    table_info = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "top_keywords": extract_top_keywords(df),
        "dataset_type": dataset_type
    }
    return table_name, table_info

# Main function to gather and update table summaries
def generate_table_summaries():
    # Load existing summaries if they exist
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            summaries = json.load(f)
    else:
        summaries = {}

    # Process files in DATA_DIR
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".csv"):
            file_path = os.path.join(DATA_DIR, filename)
            table_name, table_info = process_csv(file_path, dataset_type="original")
            # Check if table_name exists in summaries, if not, add it
            if table_name not in summaries:
                summaries[table_name] = table_info

    # Process files in UPLOAD_DIR
    for filename in os.listdir(UPLOAD_DIR):
        if filename.endswith(".csv"):
            file_path = os.path.join(UPLOAD_DIR, filename)
            table_name, table_info = process_csv(file_path, dataset_type="uploaded")
            # Check if table_name exists in summaries, if not, add it
            if table_name not in summaries:
                summaries[table_name] = table_info

    # Save updated summaries to a JSON file in the desired format
    with open(OUTPUT_FILE, "w") as f:
        json.dump(summaries, f, indent=4)

    print(f"Table summaries updated and saved to {OUTPUT_FILE}")

# Run the function to generate summaries
if __name__ == "__main__":
    generate_table_summaries()
