from usearch.io import load_matrix
from usearch.index import Index
import numpy as np

def load_embs(fbin_path: str):
    return load_matrix(fbin_path)

def eval_similarity(x, y):
    index = Index(ndim=768, metric="cosine")
    index.add(np.arange(len(x)), x)
    matches1 = index.search(y, 1)
    matches10 = index.search(y, 10)
    recall1 = np.sum(np.arange(len(x)) == matches1.keys.flatten()) / np.arange(len(x)).size
    recall10 = np.sum(np.any(matches10.keys == np.arange(len(x)).reshape(-1, 1), axis=1)) / np.arange(len(x)).size
    print(f"Recall@1: {recall1}")
    print(f"Recall@10: {recall10}")

if __name__ == "__main__":
    CODE_PATH = "/home/michael/MongooseMiner/data/validation_code.fbin"
    DOCS_PATH = "/home/michael/MongooseMiner/data/validation_docs.fbin"
    code_embs = load_embs(CODE_PATH)
    doc_embs = load_embs(DOCS_PATH)
    print("Code <-> Doc")
    eval_similarity(code_embs, doc_embs)
    print("Doc <-> Code")
    eval_similarity(doc_embs, code_embs)
    print("Done!")