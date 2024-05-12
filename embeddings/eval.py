from usearch.io import load_matrix
from usearch.index import Index
import numpy as np

def eval_similarity(x, y):
    index = Index(ndim=768, metric="cos")
    index.add(np.arange(len(x)), x)
    matches1 = index.search(y, 1)
    matches10 = index.search(y, 10)
    recall1 = np.sum(np.arange(len(x)) == matches1.keys.flatten()) / np.arange(len(x)).size
    recall10 = np.sum(np.any(matches10.keys == np.arange(len(x)).reshape(-1, 1), axis=1)) / np.arange(len(x)).size
    print(f"Recall@1: {recall1}")
    print(f"Recall@10: {recall10}")

if __name__ == "__main__":
    CODE_PATH = "/home/michael/MongooseMiner/data/test_code.fbin"
    DOCS_PATH = "/home/michael/MongooseMiner/data/test_docs.fbin"
    code_embs = load_matrix(CODE_PATH)
    doc_embs = load_matrix(DOCS_PATH)
    print("Code <-> Doc")
    eval_similarity(code_embs, doc_embs)
    print("Doc <-> Code")
    eval_similarity(doc_embs, code_embs)
    print("Done!")