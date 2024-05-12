import polars as pl
import numpy as np
from usearch.io import save_matrix
import requests
from datasets import load_dataset
from pathlib import Path

def get_dataset():
    dataset = load_dataset("unum-cloud/ann-codesearch-4m")
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    test_set = dataset["test"]
    return train_set, validation_set, test_set

def get_embedding_vecs(dataset):
    embs = dataset.filter(lambda x: len(x["func_code_string"]) < 10000 and len(x["func_documentation_string"]) < 10000).map(query_embs,  batched=True, batch_size=32, num_proc=16)
    code_vecs = embs["embed_func_code"]
    docs_vecs = embs["embed_func_doc"]
    return np.array(code_vecs), np.array(docs_vecs)

def embed(text: list[str]) -> list[np.array]:
    for _ in range(3):
        try:
            api_url = "https://cvllama3hackathon-infinity.hf.space/embeddings"
            response = requests.post(api_url, json={"model":"code-embed","input": text})
            embs = [emb["embedding"] for emb in response.json()["data"]]
            return embs
        except Exception as e:
            continue

def query_embs(list_rows: list[dict]):
    list_rows["embed_func_code"] = embed(list_rows["func_code_string"])
    list_rows["embed_func_doc"] = embed(list_rows["func_documentation_string"])
    return list_rows

if __name__ == "__main__":
    OUT_PATH = "/home/michael/MongooseMiner/data/"
    datasets = get_dataset()
    for dataset in datasets:
        if dataset.split._name == "train":
            code_vecs, docs_vecs = get_embedding_vecs(dataset)
            code_path = Path(OUT_PATH).joinpath(dataset.split._name + "_code.fbin").as_posix()
            docs_path = Path(OUT_PATH).joinpath(dataset.split._name + "_docs.fbin").as_posix()
            save_matrix(code_vecs, code_path)
            save_matrix(docs_vecs, docs_path)
    print("Done!")