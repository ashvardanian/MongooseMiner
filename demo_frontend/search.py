from infinity_emb import AsyncEmbeddingEngine, EngineArgs
import numpy as np
from usearch.index import Index, Matches
import asyncio
import pandas as pd

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
            """def HttpClient( host: str = "localhost", port: int = 8000, ssl: bool = False, headers: Optional[Dict[str, str]] = None, settings: Optional[Settings] = None, tenant: str = DEFAULT_TENANT, database: str = DEFAULT_DATABASE, ) -> ClientAPI:  Creates a client that connects to a remote Chroma server. This supports many clients connecting to the same server, and is the recommended way to use Chroma in production. Args: host: The hostname of the Chroma server. Defaults to "localhost". port: The port of the Chroma server. Defaults to "8000". ssl: Whether to use SSL to connect to the Chroma server. Defaults to False. headers: A dictionary of headers to send to the Chroma server. Defaults to {}. settings: A dictionary of settings to communicate with the chroma server. tenant: The tenant to use for this client. Defaults to the default tenant. database: The database to use for this client. Defaults to the default database.  if settings is None: settings = Settings() # Make sure paramaters are the correct types -- users can pass anything. host = str(host) port = int(port) ssl = bool(ssl) tenant = str(tenant) database = str(database) settings.chroma_api_impl = "chromadb.api.fastapi.FastAPI" if settings.chroma_server_host and settings.chroma_server_host != host: raise ValueError( f"Chroma server host provided in settings[{settings.chroma_server_host}] is different to the one provided in HttpClient: [{host}]" ) settings.chroma_server_host = host if settings.chroma_server_http_port and settings.chroma_server_http_port != port: raise ValueError( f"Chroma server http port provided in settings[{settings.chroma_server_http_port}] is different to the one provided in HttpClient: [{port}]" ) settings.chroma_server_http_port = port settings.chroma_server_ssl_enabled = ssl settings.chroma_server_headers = headers return ClientCreator(tenant=tenant, database=database, settings=settings) """,
            """def PersistentClient( path: str = "./chroma", settings: Optional[Settings] = None, tenant: str = DEFAULT_TENANT, database: str = DEFAULT_DATABASE, ) -> ClientAPI:  Creates a persistent instance of Chroma that saves to disk. This is useful for testing and development, but not recommended for production use. Args: path: The directory to save Chroma's data to. Defaults to "./chroma". tenant: The tenant to use for this client. Defaults to the default tenant. database: The database to use for this client. Defaults to the default database.  if settings is None: settings = Settings() settings.persist_directory = path settings.is_persistent = True # Make sure paramaters are the correct types -- users can pass anything. tenant = str(tenant) database = str(database) return ClientCreator(tenant=tenant, database=database, settings=settings) """,
            """class TokenTransportHeader(Enum):  Accceptable token transport headers.  # I don't love having this enum here -- it's weird to have an enum # for just two values and it's weird to have users pass X_CHROMA_TOKEN # to configure x-chroma-token. But I also like having a single source # of truth, so 🤷🏻‍♂️ AUTHORIZATION = "Authorization" X_CHROMA_TOKEN = "X-Chroma-Token""",
            "torch.sub(input, other, *, alpha=1, out=None) → TensorSubtracts other, scaled by alpha, from input.outi=inputi−alpha×otheriouti​=inputi​−alpha×otheri​Supports broadcasting to a common shape, type promotion, and integer, float, and complex inputs.Parametersinput (Tensor) – the input tensor.other (Tensor or Number) – the tensor or number to subtract from input.Keyword Argumentsalpha (Number) – the multiplier for other.out (Tensor, optional) – the output tensor.",
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
