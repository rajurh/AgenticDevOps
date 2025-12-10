import os
import asyncio
from typing import List, Dict, Any
import httpx

from error_utils import logger, log_exception


class AzureOpenAIClient:
    """Lightweight client for Azure OpenAI-style endpoints.

    The endpoint URLs should be the full deployment URLs provided by Azure, e.g.
    - Embeddings: https://<instance>.openai.azure.com/openai/deployments/<deployment>/embeddings?api-version=2023-05-15
    - Chat: https://<instance>.openai.azure.com/openai/deployments/<deployment>/chat/completions?api-version=2025-01-01-preview
    """
    def __init__(self, embedding_url: str, chat_url: str, api_key: str, timeout: int = 30):
        self.embedding_url = embedding_url
        self.chat_url = chat_url
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=timeout)

    async def embed_text(self, text: str) -> List[float]:
        if not self.embedding_url:
            raise RuntimeError("Embedding URL is not configured (AZURE_OPENAI_EMBEDDING_URL).")
        payload = {"input": text}
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        try:
            resp = await self._client.post(self.embedding_url, json=payload, headers=headers)
            resp.raise_for_status()
            body = resp.json()
            # follow Azure OpenAI embedding response shape: {data: [{embedding: [...]}], ...}
            emb = body.get("data", [None])[0]
            if not emb or "embedding" not in emb:
                raise RuntimeError("Unexpected embedding response shape from Azure OpenAI")
            return emb["embedding"]
        except httpx.RequestError as e:
            log_exception(e, "Network error when calling Azure OpenAI embeddings")
            raise RuntimeError("Network error when requesting embeddings") from e
        except httpx.HTTPStatusError as e:
            log_exception(e, "HTTP error from Azure OpenAI embeddings")
            # include status code briefly
            raise RuntimeError(f"Azure OpenAI embeddings returned HTTP {e.response.status_code}") from e
        except Exception as e:
            log_exception(e, "Failed to parse embedding response")
            raise RuntimeError("Failed to get embedding from Azure OpenAI") from e

    async def chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 512, temperature: float = 0.0) -> str:
        payload = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        try:
            resp = await self._client.post(self.chat_url, json=payload, headers=headers)
            resp.raise_for_status()
            body = resp.json()
        except httpx.RequestError as e:
            log_exception(e, "Network error when calling Azure OpenAI chat completion")
            raise RuntimeError("Network error when requesting chat completion") from e
        except httpx.HTTPStatusError as e:
            log_exception(e, "HTTP error from Azure OpenAI chat completion")
            raise RuntimeError(f"Azure OpenAI chat returned HTTP {e.response.status_code}") from e
        except Exception as e:
            log_exception(e, "Failed to get chat completion response")
            raise RuntimeError("Failed to parse chat completion response") from e
        # Azure chat response shape may be {choices: [{message: {content: "..."}}], ...}
        choice = None
        try:
            choice = body.get("choices", [None])[0]
        except Exception:
            choice = None
        if not choice:
            raise RuntimeError("No choices in chat completion response")
        msg = choice.get("message") or choice.get("content") or {}
        if isinstance(msg, dict):
            return msg.get("content", "")
        return str(msg)

    async def close(self):
        await self._client.aclose()


class RAG:
    def __init__(self, client: AzureOpenAIClient, vector_store, top_k: int = 3):
        self.client = client
        self.vector_store = vector_store
        self.top_k = top_k

    async def answer(self, query: str) -> Dict[str, Any]:
        # 1) embed query (gracefully handle failures)
        q_emb = await self.client.embed_text(query)
        # 2) retrieve top_k docs
        retrieved = self.vector_store.search(q_emb, top_k=self.top_k)
        # 3) build prompt with retrieved docs
        context_texts = []
        for d in retrieved:
            context_texts.append(f"Source (score={d['score']:.3f}):\n{d['text']}")
        context_blob = "\n\n---\n\n".join(context_texts)

        system = (
            "You are a helpful assistant for DevOps and Azure cloud operations. "
            "Answer questions using ONLY the provided context about deployment, CI/CD, security, and operations. "
            "If the question is outside the scope of the provided context (e.g., unrelated topics, personal questions, general knowledge), "
            "politely respond: 'I can only answer questions related to DevOps, deployments, CI/CD pipelines, and Azure operations based on our documentation. "
            "Please ask questions within this domain.' "
            "For in-scope questions, provide short, actionable answers and cite which source(s) you used."
        )

        user_prompt = (
            f"Context:\n{context_blob}\n\nUser question: {query}\n\n"
            f"Answer the question if it relates to DevOps, CI/CD, deployment, security, or Azure operations. "
            f"If not related to these topics, politely decline. Include citations for sources used."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]

        completion = await self.client.chat_completion(messages)
        return {"answer": completion, "retrieved": retrieved}
