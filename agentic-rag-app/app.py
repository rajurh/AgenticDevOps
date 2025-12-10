import os
import glob
import json
import asyncio
from typing import List
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from error_utils import configure_logging, logger, log_exception, exception_to_dict
from rag_core import AzureOpenAIClient, RAG
from vector_store import InMemoryVectorStore

load_dotenv()  # load from .env if present

configure_logging()

EMBEDDING_URL = os.getenv("AZURE_OPENAI_EMBEDDING_URL")
CHAT_URL = os.getenv("AZURE_OPENAI_CHAT_URL")
API_KEY = os.getenv("AZURE_OPENAI_KEY")
TOP_K = int(os.getenv("TOP_K", "3"))

if not (EMBEDDING_URL and CHAT_URL and API_KEY):
    logger.warning("Azure OpenAI environment variables are not all set. Endpoints will fail if used.")

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


# Initialize components lazily
_client: AzureOpenAIClient | None = None
_rag: RAG | None = None


async def get_client() -> AzureOpenAIClient:
    global _client
    if _client is None:
        _client = AzureOpenAIClient(embedding_url=EMBEDDING_URL, chat_url=CHAT_URL, api_key=API_KEY)
    return _client


async def get_rag() -> RAG:
    global _rag
    if _rag is None:
        client = await get_client()
        vs = InMemoryVectorStore()
        try:
            # load data files and embed them
            files = sorted(glob.glob("data/*.json"))
            docs = []
            for i, fp in enumerate(files):
                try:
                    with open(fp, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                        text = data.get("text") or json.dumps(data)
                        metadata = data.get("meta", {})
                        docs.append({"id": f"doc-{i}", "text": text, "metadata": metadata})
                except Exception as e:
                    log_exception(e, f"Failed to read/parse data file {fp}")
                    continue

            # embed documents
            async def embed_many():
                out = []
                for d in docs:
                    try:
                        emb = await client.embed_text(d["text"])
                        out.append({"id": d["id"], "text": d["text"], "embedding": emb, "metadata": d.get("metadata", {})})
                    except Exception as e:
                        log_exception(e, f"Failed to embed document id={d.get('id')}")
                        continue
                return out

            embedded = await embed_many()
            vs.add_documents(embedded)
            _rag = RAG(client=client, vector_store=vs, top_k=TOP_K)
        except Exception as e:
            log_exception(e, "Failed to initialize RAG components")
            raise
    return _rag


@app.post("/api/query")
async def query_endpoint(q: QueryRequest):
    try:
        rag = await get_rag()
        result = await rag.answer(q.query)
        return JSONResponse(result)
    except Exception as e:
        # Log full details; return safe error to client
        log_exception(e, "Error while processing query request")
        return JSONResponse({"error": "Internal server error", "details": exception_to_dict(e)}, status_code=500)


@app.on_event("shutdown")
async def shutdown_event():
    global _client
    if _client:
        await _client.close()
