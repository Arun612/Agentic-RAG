# NEXUS: Architecture & Data Flow

Below is the complete architectural graph of the NEXUS multi-agent system. It shows how a user query moves through the 10-step pipeline, interacts with the three-tier Cognitive Memory Fabric, and passes through the Adversarial Verification Parliament.

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
