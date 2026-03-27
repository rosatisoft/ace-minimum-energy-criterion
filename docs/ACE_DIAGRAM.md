# ACE Pipeline Diagram

```mermaid
flowchart TD
    U[User Input] --> P[Prompt Parsing]
    P --> G[LLM Candidate Generation]

    A[Axioms] --> RC[Reference Context]
    K[Knowledge] --> RC
    P --> RC

    RC --> E1[Reference Embeddings]
    G --> E2[Candidate Embeddings]

    E1 --> S[Subspace Construction]
    E2 --> O[Origin Cost Computation]
    S --> O

    O --> D{Decision Layer}

    D -->|best score <= low threshold| R1[Answer]
    D -->|scores too close| R2[Clarify]
    D -->|best score >= high threshold| R3[Abstain]
