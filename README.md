# Advanced RAG — Step-by-Step Notebooks

A progressive, hands-on series of Jupyter notebooks that builds a production-grade Retrieval-Augmented Generation (RAG) system from scratch.  
Each notebook is self-contained and introduces one new concept on top of the previous ones.

---

## Table of Contents

| # | Notebook | Topic | Category |
|---|----------|-------|----------|
| 01 | [Data Chunking](#01-data-chunking) | Split documents into searchable pieces | Indexing |
| 02 | [Metadata Extraction](#02-metadata-extraction) | Attach structured facts to chunks | Indexing |
| 03 | [Hierarchical Parent-Child Chunking](#03-hierarchical-parent-child-chunking) | Retrieve small, return big | Indexing |
| 04 | [Vector Storage Types](#04-vector-storage-types) | FAISS vs ChromaDB | Storage |
| 05 | [BM25 Full-Text Retrieval](#05-bm25-full-text-retrieval) | Keyword search without embeddings | Retrieval |
| 06 | [Hybrid Retrieval + RRF](#06-hybrid-retrieval--rrf) | Dense + sparse search merged | Retrieval |
| 07 | [Basic RAG Pipeline](#07-basic-rag-pipeline) | End-to-end retrieve → generate | RAG Core |
| 08 | [HyDE Query Expansion](#08-hyde-query-expansion) | Embed a fake answer, not the query | Advanced RAG |
| 09 | [Cross-Encoder Reranker](#09-cross-encoder-reranker) | Two-stage retrieval for precision | Advanced RAG |
| 10 | [Self-RAG with LangGraph](#10-self-rag-with-langgraph) | LLM grades its own retrieval | Agentic RAG |
| 11 | [RAPTOR Hierarchical Summarization](#11-raptor-hierarchical-summarization) | Multi-level summary tree | Agentic RAG |
| 12 | [Multi-LLM Routing with LangGraph](#12-multi-llm-routing-with-langgraph) | Route queries to the right model | Agentic RAG |
| 13 | [CRAG — Corrective RAG](#13-crag--corrective-rag) | Grade first, generate only if good | Agentic RAG |
| 14 | [ReAct Agent with Tools](#14-react-agent-with-tools) | Reason + Act in a tool loop | Agents |
| 15 | [Multi-Agent Coordination](#15-multi-agent-coordination) | Supervisor + specialist agents | Agents |
| 16 | [Semantic Cache](#16-semantic-cache) | Skip LLM calls for similar questions | Optimization |
| 17 | [RAGAS Evaluation](#17-ragas-evaluation) | Automated RAG quality metrics | Evaluation |
| 18 | [Hallucination Detection](#18-hallucination-detection) | Catch LLM fabrications via NLI | Evaluation |

---

## Setup

### 1. Clone and enter the project

```bash
git clone <repo-url>
cd advance_rag_exec_steps
```

### 2. Install dependencies

```bash
pip install langchain langchain-community langchain-core langchain-groq \
            langchain-huggingface langchain-chroma langchain-text-splitters \
            langgraph chromadb sentence-transformers rank-bm25 \
            ragas datasets pydantic numpy pandas python-dotenv
```

### 3. Set your API key

Create a `.env` file in the project root (already present if cloned):

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

### 4. Launch Jupyter

```bash
jupyter notebook notebooks/
```

---

## Datasets

All source documents live in `datasets/`. They are plain `.txt` files covering AI topics used as the knowledge base throughout the notebooks.

```
datasets/
├── ai.txt
├── machine_learning.txt
├── deep_learning.txt
├── nlp.txt
├── natural_language_processing.txt
├── rag.txt
├── retrieval_augmented_generation.txt
└── basic_rag_index/          ← pre-built FAISS index (notebook 007)
```

---

## Notebook Details

### Phase 1 — Indexing

#### 01. Data Chunking
`notebooks/001_data_chunking.ipynb` · [Theory](notebooks/001_data_chunking.txt)

Breaks large documents into small pieces the LLM can process. Covers `chunk_size`, `chunk_overlap`, and separator strategies. Wrong chunking is the most common cause of bad RAG answers.

**Key concepts:** `chunk_size`, `chunk_overlap`, `separator`, corpus

---

#### 02. Metadata Extraction
`notebooks/002_metadata_extraction.ipynb` · [Theory](notebooks/002_metadata_extraction.txt)

Attaches structured facts (who, when, topic, section) to each chunk using LLM-generated JSON via Pydantic schemas. Metadata enables pre- and post-retrieval filtering.

**Key concepts:** NER, structured output, Pydantic, metadata filter

---

#### 03. Hierarchical Parent-Child Chunking
`notebooks/003_hierarchical_parent_child_chunking.ipynb` · [Theory](notebooks/003_hierarchical_parent_child_chunking.txt)

Solves the precision vs. context trade-off: index small chunks for search, but return their large parent chunk to the LLM. *"Retrieve small. Return big."*

**Key concepts:** parent chunk, child chunk, docstore, MultiVectorRetriever

---

### Phase 2 — Storage & Retrieval

#### 04. Vector Storage Types
`notebooks/004_vector_storage_types.ipynb` · [Theory](notebooks/004_vector_storage_types.txt)

Compares FAISS (in-memory, fast ANN) and ChromaDB (persistent, filterable). Explains embeddings, cosine similarity, and when to choose each store.

**Key concepts:** embedding, cosine similarity, ANN, FAISS, ChromaDB, collection

---

#### 05. BM25 Full-Text Retrieval
`notebooks/005_bm25_fulltext_retrieval.ipynb` · [Theory](notebooks/005_bm25_fulltext_retrieval.txt)

Implements keyword-based retrieval with BM25 — the algorithm inside Elasticsearch. No embeddings needed. Excels at rare-word and exact-match queries.

**Key concepts:** sparse retrieval, TF-IDF, BM25, `rank_bm25`

---

#### 06. Hybrid Retrieval + RRF
`notebooks/006_hybrid_retrieval_rrf.ipynb` · [Theory](notebooks/006_hybrid_retrieval_rrf.txt)

Runs dense (embedding) and sparse (BM25) search in parallel, then merges results with Reciprocal Rank Fusion. Beats either method alone on most benchmarks.

**Key concepts:** hybrid retrieval, RRF, EnsembleRetriever, rank fusion

---

### Phase 3 — RAG Core

#### 07. Basic RAG Pipeline
`notebooks/007_basic_rag.ipynb` · [Theory](notebooks/007_basic_rag.txt)

The complete baseline: embed documents → store in FAISS → retrieve on query → stuff into prompt → generate answer with Groq LLM. The foundation every later notebook extends.

**Key concepts:** RAG, retriever, generator, context, chain, grounding

---

### Phase 4 — Advanced RAG

#### 08. HyDE Query Expansion
`notebooks/008_hyde_query_expansion.ipynb` · [Theory](notebooks/008_hyde_query_expansion.txt)

Fixes the query-document embedding gap. The LLM writes a hypothetical answer first; that answer is embedded and used for retrieval instead of the raw query — much closer to document-space.

**Key concepts:** HyDE, hypothetical document, query expansion, embedding gap

---

#### 09. Cross-Encoder Reranker
`notebooks/009_cross_encoder_reranker.ipynb` · [Theory](notebooks/009_cross_encoder_reranker.txt)

Two-stage retrieval: fast bi-encoder search fetches top-50 candidates, then a precise cross-encoder reads query+document together to rerank and return only the best results.

**Key concepts:** bi-encoder, cross-encoder, reranking, two-stage retrieval

---

### Phase 5 — Agentic RAG

#### 10. Self-RAG with LangGraph
`notebooks/010_self_rag_langgraph.ipynb` · [Theory](notebooks/010_self_rag_langgraph.txt)

The LLM grades its own retrieval and generation in a loop: are these docs relevant? Is my answer grounded? If not — retry. Turns RAG into a self-correcting state machine.

**Key concepts:** Self-RAG, relevance grade, hallucination check, LangGraph, state, node

---

#### 11. RAPTOR Hierarchical Summarization
`notebooks/011_raptor_hierarchical_summarization.ipynb` · [Theory](notebooks/011_raptor_hierarchical_summarization.txt)

Builds a multi-level summary tree over the document corpus. Indexes both leaf chunks and summaries so broad questions hit high-level summaries and specific questions hit leaf chunks.

**Key concepts:** RAPTOR, leaf chunk, cluster, summary tree, multi-level retrieval

---

#### 12. Multi-LLM Routing with LangGraph
`notebooks/012_multi_llm_routing_langgraph.ipynb` · [Theory](notebooks/012_multi_llm_routing_langgraph.txt)

Classifies each query and routes it to the cheapest model that can answer it. Simple → fast cheap model. Hard → powerful expensive model. Saves 3-5× cost at the same quality.

**Key concepts:** routing, classifier, simple/complex/code query, token cost, LangGraph

---

#### 13. CRAG — Corrective RAG
`notebooks/013_crag_corrective_rag.ipynb` · [Theory](notebooks/013_crag_corrective_rag.txt)

Grades retrieved documents *before* generating. Correct docs → use them. Ambiguous → supplement with web search. Incorrect → discard and search the web. Never generates from bad context.

**Key concepts:** CRAG, relevance grade, web search fallback, knowledge refinement

---

### Phase 6 — Agents

#### 14. ReAct Agent with Tools
`notebooks/014_react_agent_tools_langgraph.ipynb` · [Theory](notebooks/014_react_agent_tools_langgraph.txt)

Implements the Reason + Act loop: the LLM thinks, calls a tool, observes the result, and repeats until it has a final answer. Tools include RAG search, calculator, and web lookup.

**Key concepts:** ReAct, tool, tool call, observation, `@tool` decorator, tool schema

---

#### 15. Multi-Agent Coordination
`notebooks/015_multi_agent_coordination_langgraph.ipynb` · [Theory](notebooks/015_multi_agent_coordination_langgraph.txt)

Multiple specialized LLM agents (ML expert, NLP expert, …) managed by a supervisor that routes the query and a synthesizer that combines the specialist answers into a final response.

**Key concepts:** multi-agent, supervisor, specialist, synthesizer, orchestration, domain

---

### Phase 7 — Optimization & Evaluation

#### 16. Semantic Cache
`notebooks/016_semantic_cache.ipynb` · [Theory](notebooks/016_semantic_cache.txt)

Caches LLM responses and matches incoming queries by meaning (cosine similarity), not exact string. *"What is ML?"* and *"Define machine learning"* return the same cached answer.

**Key concepts:** semantic cache, cache hit/miss, cosine threshold, TTL

---

#### 17. RAGAS Evaluation
`notebooks/017_ragas_evaluation.ipynb` · [Theory](notebooks/017_ragas_evaluation.txt)

Automatically scores the RAG pipeline on faithfulness, answer relevancy, context precision, and context recall. Treats evaluation like a unit test — run on every change.

**Key concepts:** RAGAS, eval dataset, ground truth, faithfulness, answer relevancy, LLM-as-judge

---

#### 18. Hallucination Detection
`notebooks/018_hallucination_detection.ipynb` · [Theory](notebooks/018_hallucination_detection.txt)

Detects when the LLM generates claims not supported by the retrieved context. Uses Natural Language Inference (NLI) to classify each sentence as entailed, contradicted, or neutral.

**Key concepts:** hallucination, NLI, entailment, contradiction, grounding

---

## Recommended Study Order

```
001 → 002 → 003          # Build your indexing pipeline
004 → 005 → 006          # Understand storage and retrieval options
007                       # Wire up the basic end-to-end RAG
008 → 009                # Improve retrieval quality
010 → 011 → 012 → 013   # Add self-correction and routing
014 → 015                # Move to agentic architectures
016 → 017 → 018          # Optimize and evaluate
```

---

## Tech Stack

| Library | Role |
|---------|------|
| `langchain` / `langgraph` | Chains, retrievers, agents, state graphs |
| `langchain-groq` | LLM calls via Groq API (Llama 3) |
| `langchain-huggingface` | Sentence-transformer embeddings |
| `chromadb` | Persistent vector store |
| `sentence-transformers` | `all-MiniLM-L6-v2` embedding model |
| `rank_bm25` | BM25 sparse retrieval |
| `ragas` | RAG evaluation metrics |
| `pydantic` | Structured output schemas |
| `python-dotenv` | `.env` API key loading |

---

## Project Structure

```
advance_rag_exec_steps/
├── notebooks/
│   ├── 001_data_chunking.ipynb
│   ├── 001_data_chunking.txt          ← theory/glossary companion
│   ├── ...
│   └── 018_hallucination_detection.ipynb
├── datasets/
│   ├── ai.txt
│   ├── machine_learning.txt
│   └── ...
├── .env                               ← GROQ_API_KEY (not committed)
└── README.md
```
