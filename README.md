# NEXUS — Neuro-Episodic Expert Unified System

> A Multi-Agent Agentic RAG System 

## 🚀 Quick Start

### 1. Install Dependencies
```bash
conda activate pytorch
pip install -r requirements.txt
```

### 2. Configure API Keys
Edit `.env` with your API keys:
```
GROQ_API_KEY=gsk_your-key-here
TAVILY_API_KEY=tvly-your-key-here  # Optional, for web search
```
Get a free Groq key at [console.groq.com](https://console.groq.com)

### 3. Run NEXUS
```bash
python -m nexus.main
```

### 4. Ingest Documents
```
NEXUS> /ingest path/to/your/document.pdf
NEXUS> /ingest path/to/your/notes.txt
```

### 5. Ask Questions
```
NEXUS> What are the key concepts in agentic RAG?
```

---

## 🧠 The 7 Unique Concepts

| # | Concept | What It Does | File |
|---|---------|-------------|------|
| 1 | **Cognitive Memory Fabric (CMF)** | 3-tier memory: episodic + semantic + procedural | `memory.py` |
| 2 | **Adversarial Verification Parliament (AVP)** | 3 agents debate document quality before answering | `agents.py` |
| 3 | **Speculative Parallel Retrieval + Fusion (SPRF)** | Vector + Keyword + Web retrieval fused via RRF | `agents.py` |
| 4 | **Graph-of-Thought Query Decomposition (GoTQD)** | Complex queries become parallel sub-query DAGs | `agents.py` |
| 5 | **Meta-Learning Retrieval Optimizer (MLRO)** | System learns best strategies over time | `memory.py` |
| 6 | **Epistemic Uncertainty Quantification (EUQ)** | Multi-dimensional confidence scoring with flags | `agents.py` |
| 7 | **Self-Evolving Tool Synthesis (SETS)** | Dynamic Python tool generation for computation | `agents.py` |

---

## 📁 Project Structure

```
nexus/
├── __init__.py    # Package marker
├── config.py      # Configuration, LLM client (Groq), embeddings, CLI formatters
├── memory.py      # Episodic + Semantic (knowledge graph) + Procedural memory + Consolidator
├── agents.py      # All 11 agents: planner, 3 retrievers, RRF fusion, parliament, synthesis, confidence, tool smith
├── core.py        # Message bus + Orchestrator (coordinates the 10-step pipeline)
└── main.py        # CLI entry point, document ingestion, user feedback loop
```

---

## 🏗️ System Architecture & Data Flow

```mermaid
graph TD
    %% Styling
    classDef user fill:#2d3436,stroke:#dfe6e9,stroke-width:2px,color:#fff
    classDef memory fill:#0984e3,stroke:#74b9ff,stroke-width:2px,color:#fff
    classDef agent fill:#6c5ce7,stroke:#a29bfe,stroke-width:2px,color:#fff
    classDef retrieval fill:#00b894,stroke:#55efc4,stroke-width:2px,color:#fff
    classDef verification fill:#d63031,stroke:#ff7675,stroke-width:2px,color:#fff
    classDef output fill:#e17055,stroke:#fab1a0,stroke-width:2px,color:#fff
    classDef decision fill:#fdcb6e,stroke:#ffeaa7,stroke-width:2px,color:#2d3436

    %% Input
    UserQ["👤 User Query"]:::user --> S1

    %% 10-Step Pipeline
    subgraph Meta-Orchestrator [10-Step Query Lifecycle]
        S1["1. Check Episodic Memory<br>(Find similar past interactions)"]:::memory --> S2
        S2["2. Consult Procedural Memory<br>(MLRO recommends best strategy)"]:::memory --> S3
        S3["3. Planning Agent<br>(GoTQD: Decompose into DAG)"]:::agent --> S4
        
        %% Retrieval Block
        subgraph SPRF [4. Speculative Parallel Retrieval Fusion]
            direction LR
            V[Vector Search<br>ChromaDB]:::retrieval
            K[Keyword Search<br>BM25]:::retrieval
            W[Web Search<br>Tavily]:::retrieval
            RRF[Reciprocal Rank Fusion<br>Merge & Rerank]:::retrieval
            
            S4 --> V & K & W
            V & K & W --> RRF
        end
        
        RRF --> S5
        
        %% Parliament Block
        subgraph AVP [5. Adversarial Verification Parliament]
            direction LR
            Advocate[Advocate Agent<br>Argues FOR]:::verification
            Prosecutor[Prosecutor Agent<br>Argues AGAINST]:::verification
            Judge[Judge Agent<br>Weighs Evidence]:::verification
            
            S5 --> Advocate & Prosecutor
            Advocate & Prosecutor --> Judge
        end
        
        Judge --> S6{"Verdict?"}:::decision
        
        %% Verdict Routing
        S6 -->|REJECT| Refine["Refine Query & Re-retrieve"]:::agent
        Refine --> SPRF
        
        S6 -->|ACCEPT / PARTIAL| S7
        
        %% Synthesis & Output
        S7["6. Tool Smith Agent<br>(Dynamic Python execution if needed)"]:::agent --> S8
        S8["7. Synthesis Agent<br>(Draft answer with citations)"]:::agent --> S9
        S9["8. Confidence Agent<br>(EUQ: Score accuracy & coverage)"]:::agent --> S10
        S10["9. Final Output<br>(Answer + Confidence Card)"]:::output --> S11
        S11["10. Update Memory Fabric<br>(Log Episode, Learn Strategy, Update Graph)"]:::memory
    end

    %% Memory Updates
    S11 -.->|Updates| EpisodicDB[(Episodic Memory)]
    S11 -.->|Extracts entities| SemanticDB[(Semantic Knowledge Graph)]
    S11 -.->|Updates success rate| ProceduralDB[(Procedural MLRO)]
```

---

## ⚙️ 10-Step Query Pipeline

```
1. Check Memory        → Find similar past queries in episodic memory
2. Get Strategy        → Procedural memory recommends best approach (MLRO)
3. Plan Sub-queries    → GoTQD decomposes complex queries into a DAG
4. Parallel Retrieval  → Vector (ChromaDB) + Keyword (BM25) + Web (Tavily) → RRF Fusion
5. Verification        → Advocate argues FOR, Prosecutor argues AGAINST, Judge renders verdict
6. Tool Smith          → Generate computation tools if needed (SETS)
7. Synthesize Answer   → Generate cited answer from verified sources
8. Confidence Scoring  → 4-dimensional scoring: accuracy, agreement, recency, coverage (EUQ)
9. Display Results     → Rich formatted answer + confidence card + source table
10. Update Memory      → Record episode, update knowledge graph, learn strategy outcomes
```

---

## 🔧 CLI Commands

| Command | Description |
|---------|-------------|
| `/ingest <file>` | Add documents (PDF/text) to knowledge base |
| `/stats` | View system statistics & performance matrix |
| `/memory` | Check memory system status |
| `/clear` | Reset all memory |
| `/help` | Show all commands |
| `/quit` | Exit NEXUS |

---

## 📦 Tech Stack

| Library | Purpose |
|---------|---------|
| `openai` | Groq API client (OpenAI-compatible) |
| `sentence-transformers` | Local embeddings (all-MiniLM-L6-v2) |
| `chromadb` | Vector database for semantic search |
| `rank-bm25` | BM25 keyword search |
| `tavily-python` | Web search API |
| `networkx` | Knowledge graph (semantic memory) |
| `rich` | Terminal UI |
| `pypdf` | PDF ingestion |

---

## 🔑 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Groq API key (free tier) | ✅ |
| `GROQ_MODEL` | Model name (default: `llama-3.3-70b-versatile`) | ❌ |
| `TAVILY_API_KEY` | Tavily web search key | ❌ |
| `EMBEDDING_PROVIDER` | `local` (default) | ❌ |
| `MAX_RETRIEVAL_RESULTS` | Top-k results per retriever (default: 5) | ❌ |
| `CONFIDENCE_THRESHOLD` | Min confidence threshold (default: 0.6) | ❌ |
| `ENABLE_WEB_SEARCH` | Enable Tavily web search (default: true) | ❌ |
| `ENABLE_TOOL_SYNTHESIS` | Enable dynamic tool generation (default: true) | ❌ |
| `LOG_LEVEL` | Logging level (default: INFO) | ❌ |
