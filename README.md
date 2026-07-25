# LangChain + Ollama RAG Playground

A local, offline RAG (Retrieval-Augmented Generation) pipeline built with LangChain and Ollama — no API keys, no external calls. Built as a hands-on learning project covering LangChain's core components: chat models, prompt templates, few-shot example selection, structured output parsing, chains, and agents.

## Stack

- **Chat model:** `mistral` (via [Ollama](https://ollama.com))
- **Embeddings:** `nomic-embed-text` (via Ollama)
- **Vector store:** FAISS (local, in-memory)
- **Structured output:** Pydantic + `PydanticOutputParser`
- **Framework:** LangChain (`langchain-core`, `langchain-community`, `langchain-ollama`, `langchain-classic`)

## Prerequisites

1. **[Ollama](https://ollama.com/download)** installed and running.
2. Pull the two models this project uses:
   ```powershell
   ollama pull mistral
   ollama pull nomic-embed-text
   ```
3. Confirm they're available:
   ```powershell
   ollama list
   ```

## Setup

```powershell
# Clone the repo
git clone <your-repo-url>
cd <repo-folder>

# Create and activate a virtual environment
python -m venv env
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& .\env\Scripts\Activate.ps1)

# Install dependencies
pip install -r requirements.txt
```

## Running the pipeline

```powershell
python rag_chain_local.py
```

This runs a minimal RAG chain: a query flows through a `ChatPromptTemplate` (with a `SemanticSimilarityExampleSelector` picking relevant few-shot examples), gets answered by `mistral` via `ChatOllama`, and the raw response is parsed into a structured `RAGAnswer` object (`answer` + `sources`) via `PydanticOutputParser`.

## Notebook exploration

`langchain_playground.ipynb` breaks the same concepts down cell-by-cell for experimentation — language models vs. chat models, chat messages, prompt templates, output parsers, chains, documents, and agents.

**Important:** make sure VS Code's selected kernel points at `.\env\Scripts\python.exe`, not a global Python install, or you'll hit `ModuleNotFoundError` even with everything installed correctly. Check with:
```python
import sys
print(sys.executable)
```

## Known quirks with local models

- Mistral 7B is far less reliable than GPT-4/Claude at strictly following output format instructions (JSON schemas, ReAct-style `Action:`/`Final Answer:` formats for agents). The prompt in `rag_chain_local.py` repeats the JSON instruction in both the system and human turns to improve compliance.
- For agents specifically, pass `handle_parsing_errors=True` to `AgentExecutor`/`initialize_agent` so a malformed response gets fed back to the model for a retry instead of crashing the run.
- CPU-only inference (no discrete GPU) means each call — especially the embedding step inside the example selector — can be noticeably slower than a hosted API. Worth testing single queries before scaling up example banks or document counts.

## Project structure

```
.
├── rag_chain_local.py       # working end-to-end RAG chain
├── langchain_playground.ipynb  # concept-by-concept notebook
├── requirements.txt
└── README.md
```

## Roadmap

- [ ] Swap placeholder `{context}` string for real document retrieval (PDF/text loaders → chunking → FAISS index)
- [ ] Add `OutputFixingParser` or a manual retry loop for parser failures
- [ ] Explore `create_react_agent` (LangGraph) as a more robust alternative to `ZERO_SHOT_REACT_DESCRIPTION` for local models