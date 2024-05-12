from infinity_emb import AsyncEmbeddingEngine, EngineArgs
import numpy as np
from usearch.index import Index, Matches
import asyncio
import pandas as pd
import os
os.environ["HF_HOME"] = "/app"
os.environ["TRANSFORMERS_CACHE"] = "/app"
os.environ["INFINITY_QUEUE_SIZE"] = "512000"

engine = AsyncEmbeddingEngine.from_args(
    EngineArgs(
        model_name_or_path="michaelfeil/jina-embeddings-v2-base-code",
        batch_size=8,     
    )
)


async def embed_texts(texts: list[str]) -> np.ndarray:
    async with engine:
        embeddings = (await engine.embed(texts))[0]
        return np.array(embeddings)

def embed_texts_sync(texts: list[str]) -> np.ndarray:
    loop =  asyncio.new_event_loop()
    return loop.run_until_complete(embed_texts(texts))

index = None
docs_index = None


def build_index(demo_mode=True):
    global index, docs_index
    index = Index(
        ndim=embed_texts_sync(["Hi"]).shape[
            -1
        ],  # Define the number of dimensions in input vectors
        metric="cos",  # Choose 'l2sq', 'haversine' or other metric, default = 'ip'
        dtype="f16",  # Quantize to 'f16' or 'i8' if needed, default = 'f32'
        connectivity=16,  # How frequent should the connections in the graph be, optional
        expansion_add=128,  # Control the recall of indexing, optional
        expansion_search=64,  # Control the quality of search, optional
    )
    if demo_mode:
        docs_index = [
            "torch.add(*demo)",
            "torch.mul(*demo)",
            "torch.div(*demo)",
            "torch.sub(*demo)",
        ]
        embeddings = embed_texts_sync(docs_index)
        index.add(np.arange(len(docs_index)), embeddings)
        return
    # TODO: Michael, load parquet with embeddings


if index is None:
    build_index()


def answer_query(query: str) -> list[str]:
    embedding = embed_texts_sync([query])
    matches = index.search(embedding, 10)
    texts = [docs_index[match.key] for match in matches]
    return texts


if __name__ == "__main__":
    print(answer_query("torch.mul(*demo2)"))
