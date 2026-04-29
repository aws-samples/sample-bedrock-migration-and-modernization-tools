# Haystack Pipeline Architecture Diagram

```mermaid
graph TD
    %% Data Ingestion Pipeline
    A[PDF Documents<br/>AMZN-2023-10k.pdf] --> B[PyPDFToDocument<br/>Converter]
    B --> C[DocumentCleaner<br/>Preprocessor]
    C --> D[DocumentSplitter<br/>split_by=word, length=250]
    D --> E[AmazonAmazon BedrockDocumentEmbedder<br/>cohere.embed-english-v3]
    E --> F[DocumentWriter]
    F --> G[(InMemoryDocumentStore<br/>Vector Database)]
    
    %% Agent Pipeline
    H[User Query] --> I[Financial Research Agent<br/>Nova Premier v1.0]
    G --> J[InMemoryBM25Retriever<br/>RAG Tool]
    K[DuckDuckGo WebSearch<br/>Web Tool] 
    
    J --> I
    K --> I
    
    I --> L[Agent Response<br/>with Tool Calls]
    
    %% Planned Evaluation Pipeline
    M[Test Dataset<br/>Questions & Expected Answers] --> N{Evaluation Pipeline}
    L --> N
    
    N --> O[Semantic Answer Similarity<br/>Evaluator]
    N --> P[LLM Evaluator<br/>Reasoning & Tool Choice]
    N --> Q[Context Relevance<br/>Evaluator]
    
    %% Evaluation Outputs
    O --> R[Final Answer Quality<br/>Score]
    P --> S[Tool Selection &<br/>Reasoning Quality Score]
    Q --> T[Retrieval Quality<br/>Score]
    
    R --> U[Combined Evaluation<br/>Report]
    S --> U
    T --> U
    
    %% Component Details
    subgraph "Current Implementation"
        B
        C
        D
        E
        F
        G
        I
        J
        K
    end
    
    subgraph "Planned Evaluation Components"
        O
        P
        Q
        U
    end
    
    %% Styling
    classDef haystack fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef evaluation fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef data fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    
    class B,C,D,E,F,G,I,J,K haystack
    class O,P,Q,U evaluation
    class A,M,H data
```

## Diagram Overview

This diagram shows your complete Haystack pipeline architecture:

### Current Implementation (Blue Components)
- **Document Ingestion**: PDF → Converter → Cleaner → Splitter → Embedder → Vector Store
- **Agent Pipeline**: User query → Agent with RAG + Web Search tools → Response

### Planned Evaluation Components (Purple Components)
- **Semantic Answer Similarity**: Evaluates final answer quality
- **LLM Evaluator**: Assesses reasoning and tool choice correctness
- **Context Relevance**: Measures retrieval quality

### Key Features
- Color-coded components by type (Haystack/Evaluation/Data)
- Shows data flow and dependencies
- Highlights both current and planned components
- Demonstrates end-to-end evaluation strategy