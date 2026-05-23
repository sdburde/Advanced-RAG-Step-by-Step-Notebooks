# Advanced RAG — Step-by-Step Notebooks

A progressive series of Jupyter notebooks building a production-grade RAG system from scratch. Each notebook is self-contained and adds one concept on top of the previous.

---

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your [Groq API key](https://console.groq.com):

```bash
cp .env.example .env
```

---

## Notebooks

| # | File | Topic |
|---|------|-------|
| 01 | `001_data_loaders.ipynb` | Load text, PDF, JSON, YouTube transcripts |
| 02 | `002_data_chunking.ipynb` | Split documents into searchable chunks |
| 03 | `003_metadata_extraction.ipynb` | Attach structured facts to chunks |
| 04 | `004_hierarchical_parent_child_chunking.ipynb` | Retrieve small, return big |
| 05 | `005_proposition_chunking.ipynb` | Atomic factual statements as index units |
| 06 | `006_late_chunking.ipynb` | Embed full doc first, chunk after |
| 07 | `007_vector_storage_types.ipynb` | FAISS vs ChromaDB |
| 08 | `008_bm25_fulltext_retrieval.ipynb` | Keyword search without embeddings |
| 09 | `009_vectorless_rag.ipynb` | Full RAG with no embedding model |
| 10 | `010_basic_rag.ipynb` | End-to-end retrieve → generate pipeline |
| 11 | `011_hybrid_retrieval_rrf.ipynb` | Dense + sparse search merged with RRF |
| 12 | `012_cross_encoder_reranker.ipynb` | Two-stage retrieval for precision |
| 13 | `013_hyde_query_expansion.ipynb` | Embed a fake answer, not the query |
| 14 | `014_multi_query_rag.ipynb` | Generate N query variants, merge results |
| 15 | `015_self_querying_rag.ipynb` | LLM decomposes query into semantic + filter |
| 16 | `016_contextual_retrieval.ipynb` | Prepend LLM-written context to each chunk |
| 17 | `017_self_rag_langgraph.ipynb` | LLM grades its own retrieval in a loop |
| 18 | `018_crag_corrective_rag.ipynb` | Grade first, generate only if docs are good |
| 19 | `019_raptor_hierarchical_summarization.ipynb` | Multi-level summary tree |
| 20 | `020_graphrag_entities.ipynb` | Knowledge graph as retrieval source |
| 21 | `021_sql_rag.ipynb` | NL → SQL → answer over structured data |
| 22 | `022_speculative_rag.ipynb` | Small LLM drafts in parallel, large LLM synthesizes |
| 23 | `023_react_agent_tools_langgraph.ipynb` | Reason + Act tool loop |
| 24 | `024_multi_llm_routing_langgraph.ipynb` | Route queries to the right model |
| 25 | `025_agentic_rag_planning.ipynb` | LLM decides when and what to retrieve |
| 26 | `026_multi_agent_coordination_langgraph.ipynb` | Supervisor + specialist agents |
| 27 | `027_semantic_cache.ipynb` | Skip LLM calls for similar questions |
| 28 | `028_ragas_evaluation.ipynb` | Automated RAG quality metrics |
| 29 | `029_hallucination_detection.ipynb` | Detect LLM fabrications via NLI |

Each notebook has a companion `.txt` file with theory and glossary.

---

## Datasets

```
datasets/
├── ai.txt
├── machine_learning.txt
├── deep_learning.txt
├── nlp.txt / natural_language_processing.txt
├── rag.txt / retrieval_augmented_generation.txt
├── NIPS-2017-attention-is-all-you-need-Paper.pdf
├── rag_qa_dataset.json
├── yt_transcript_sample.json
└── ai_papers.db                ← SQLite (used by notebook 021)
```

---

## Structure

```
advance_rag_exec_steps/
├── rag_notebooks/      # Jupyter notebooks + .txt theory files
├── datasets/           # Source documents and knowledge bases
├── requirements.txt    # Pinned dependencies
├── .env.example        # API key template
└── README.md
```
