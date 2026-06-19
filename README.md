# Retrieval-Augmented Question Answering System

A modular Retrieval-Augmented Generation (RAG) application built with Flask, Ollama, and local vector search.

The system answers questions against a custom document corpus using semantic retrieval, hybrid ranking strategies, and grounded LLM generation.

Unlike simple top-k retrieval systems, this project combines vector search, entity-aware retrieval, multi-hop expansion, reranking, and retrieval filtering to improve answer quality and reduce hallucinations.

## Features

### Retrieval

* Semantic search using embeddings generated with `nomic-embed-text`
* Cosine similarity vector search
* Persistent vector index generation and caching
* Automatic index invalidation when source documents change
* Query embedding cache
* Search result cache

### Hybrid Ranking

* Semantic similarity scoring
* Exact entity matching
* Keyword retrieval
* Lexical overlap scoring
* Custom reranking pipeline

### Multi-Hop Retrieval

The retrieval pipeline performs:

1. Entity-first retrieval
2. Semantic retrieval using the full query
3. Entity expansion from retrieved documents
4. Second-pass retrieval using discovered entities
5. Score fusion and reranking

This improves retrieval quality for questions involving multiple entities or relationships spread across documents.

### Generation

* Local LLM inference using Ollama
* Grounded answer generation
* Source-aware prompting
* Context aggregation across multiple documents
* Explicit hallucination prevention instructions
* Automatic retry and exponential backoff for transient LLM failures

## API

### Ask Question

**Endpoint**

```http
POST /ask
```

**Request**

```json
{
  "query": "Who worked with John on Project Atlas?",
  "k": 3
}
```

**Response**

```json
{
  "question": "...",
  "answer": "...",
  "sources": [...]
}
```

### Health Check

**Endpoint**

```http
GET /health
```

**Response**

```json
{
  "status": "ok"
}
```

## Architecture

```text
User Query
    │
    ▼
Validation Layer
    │
    ▼
Hybrid Retrieval Pipeline
    │
    ├── Vector Search
    ├── Entity Retrieval
    ├── Multi-Hop Expansion
    └── Reranking
    │
    ▼
Context Builder
    │
    ▼
Ollama (Llama 3.1)
    │
    ▼
Grounded Response
```

## Technology Stack

* Python
* Flask
* Ollama
* Llama 3.1
* nomic-embed-text
* NumPy
* Requests

## Design Goals

* Minimize hallucinations through retrieval grounding
* Improve retrieval precision through hybrid ranking
* Keep infrastructure lightweight and fully local
* Support experimentation with retrieval strategies and evaluation

## Key Technical Concepts

- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Vector Embeddings
- Cosine Similarity Ranking
- Multi-Hop Retrieval
- Hybrid Retrieval (Semantic + Keyword)
- Entity-Aware Search
- Retrieval Reranking
- Context Grounding
- Hallucination Mitigation
- LLM Retry and Resilience Patterns
- Index Persistence and Caching