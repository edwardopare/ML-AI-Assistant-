# Data Governance — RAG AI

## Overview

This document describes how document data flows through the RAG AI system,
what leaves the local environment, and the required legal and operational
review steps before adding a new data source or LLM provider.

---

## Data flow

`
PDF files (local disk)
    │
    ▼ ingest (src/ingest.py)
Text chunks + metadata
    │
    ▼ embed (src/embeddings.py — all-MiniLM-L6-v2, local)
Float32 embedding vectors
    │
    ▼ persist (ChromaDB — local .chromadb/)
Vector index + raw chunk text
    │
    ▼ retrieve (src/retrieval.py — local query)
Top-k chunks (text + metadata)
    │
    ▼ generate (src/agent.py → OpenRouter)
    │   Chunks sent as context in the HTTP request body
    │   ← LLM answer returned
    ▼
Answer + citations
`

### What leaves the machine

| Data | Destination | Conditions |
|------|------------|------------|
| Retrieved text chunks (≤ MAX_CONTEXT_CHARS chars) | OpenRouter API | Every query |
| User question | OpenRouter API | Every query |
| Conversation history (last N turns) | OpenRouter API | Multi-turn sessions |
| HTTP request metadata (IP, headers) | OpenRouter API server logs | Every request |

**Embeddings and the raw PDF files never leave the machine.**

---

## OpenRouter data policy

OpenRouter routes requests to third-party LLM providers (OpenAI, Anthropic, etc.).

- **Policy reference**: https://openrouter.ai/privacy
- **Logging**: OpenRouter may log prompt/completion pairs for abuse detection.
  Review the current policy before processing sensitive data.
- **DPA**: A Data Processing Agreement (DPA) may be required for GDPR-regulated
  or HIPAA-regulated data.  Contact OpenRouter at legal@openrouter.ai.

> [!CAUTION]
> Do NOT ingest documents containing personal health information (PHI),
> personally identifiable information (PII), or trade secrets unless you have
> completed a DPA and legal review with OpenRouter and any downstream providers.

---

## Adding a new LLM provider

Before changing OPENROUTER_MODEL to a new provider or switching to a
direct-provider API:

1. Review the provider's data retention and logging policy.
2. Verify whether a DPA is required for your data classification.
3. Update this document with the provider's policy link.
4. Get sign-off from a data steward or legal counsel if PII/PHI is involved.

---

## Document content — git rules

The .gitignore file excludes all document files under data/:

`
data/*.pdf, data/**/*.pdf
data/*.docx, data/**/*.docx
data/*.txt, data/**/*.txt
… and other document extensions
`

**Never remove or weaken these rules** without explicit review.
The data/ directory itself is tracked only via data/.gitkeep.

To add new document types, update .gitignore **and** this document.

---

## Pre-ingestion checklist

Before adding documents to data/:

- [ ] Confirm documents are cleared for cloud processing with the data owner.
- [ ] Remove or redact PII / PHI / trade secrets if present.
- [ ] Confirm the file will not be committed (verify with git status data/).
- [ ] Note the document source and access rights in an internal register.

---

## Redaction

src/security.py contains edact_messages(), which strips common
credential patterns (email addresses, bearer tokens, long hex strings, phone
numbers) from LLM request bodies before transmission.

This is a **defence-in-depth measure**, not a substitute for properly
de-identifying documents before ingestion.

---

## Contact

For questions about this policy, contact the project maintainer or your
organisation's data protection officer.
