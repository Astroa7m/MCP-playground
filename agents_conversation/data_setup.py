import json
import os
import pandas as pd

import chromadb

# vector db name
collection_name = "aou_tutor_modules_conversations"
client_file_name = "./chroma"
# info to be dir
CSV_DIR = "./csv"
# file that holds csv schema summary
SCHMEA_SUMMARY_FILE = "csv_schema_summary.json"

def flatten_row(row_dict):
    return "\n".join(f"{key}: {value}" for key, value in row_dict.items())


def create_vector_db_and_schema_summary():
    schema_summary = []

    # embedding model and chroma client (vector db)
    client = chromadb.PersistentClient(client_file_name)
    collection = client.get_or_create_collection(collection_name)

    for filename in os.listdir(CSV_DIR):
        if filename.endswith(".csv"):
            path = os.path.join(CSV_DIR, filename)
            df = pd.read_csv(path)

            # convertoing each row to key-value string
            documents = [flatten_row(row.to_dict()) for _, row in df.iterrows()]

            collection.add(
                documents=documents,
                # embeddings=embeddings,
                ids=[f"{filename}_row_{i}" for i in range(len(documents))],
                metadatas=[{
                    "source_file": filename,
                    "row_index": i,
                    "columns": str(list(df.columns))
                } for i in range(len(documents))]
            )

            schema_summary.append({
                "source_file": filename,
                "columns": list(df.columns)
            })
    print("Done creating vector db")
    with open(SCHMEA_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(schema_summary, f, indent=2)
    print("Done creating schema summary")


if __name__ == "__main__":
    create_vector_db_and_schema_summary()