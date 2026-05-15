# Assumptions & Tradeoffs

## Assumptions

### System Scale & Deployment
- **Moderate load**: Single-server deployment with <100 concurrent users expected.
- **Document volume**: <1000 documents in production initially.
- **Groq API availability**: LLM API calls are reliable; fallback mechanisms handle transient failures.
- **Network reliability**: Client-server communication is stable; long polling for status checks is acceptable.

### User & Workflow
- **Operator expertise**: Users understand legal document structure and can provide quality feedback for pattern learning.
- **Document quality**: Input documents are primarily scanned PDFs, native PDFs, or structured text; highly corrupted files are rare.
- **Batch processing acceptable**: Document processing (OCR, chunking, embedding) can run asynchronously; users can poll status.

### Technical Stack
- **Python ecosystem maturity**: PyMuPDF, Tesseract, ChromaDB, SQLAlchemy are stable and maintainable.
- **Groq API stability**: The API is available and responsive during normal operation.
- **Browser compatibility**: Client-side JavaScript is ES6+ compatible; no IE support required.
- **Local storage**: SQLite and ChromaDB persistent storage on the same machine/filesystem is acceptable.

### Language & LLM Behavior
- **English-language documents**: The system is optimized for English legal documents; multilingual support is not prioritized.
- **LLM compliance with citations**: The model generally follows `[Evidence N]` citation instructions; occasional violations are acceptable and marked as `INSUFFICIENT EVIDENCE`.
- **Pattern repeatability**: Operator edits contain generalizable improvement patterns that can be extracted semantically by the LLM.

---

## Tradeoffs

### Storage & Scalability

| Choice | Benefit | Cost |
|---|---|---|
| **SQLite** | Zero configuration, no external service required | Not suitable for >1M records or concurrent writes; requires migration to PostgreSQL for scale |
| **ChromaDB (local)** | Persistent vectors without external dependency | Less feature-rich than Pinecone/Weaviate; vector scaling is limited to available disk/memory |

**Mitigation**: Connection string and vector store are abstracted in config; migration path is clear.

---

### Retrieval & Grounding

| Choice | Benefit | Cost |
|---|---|---|
| **Sentence-aware chunking** | Preserves semantic coherence; better retrieval quality | Chunks can be longer than desired; sentence detection relies on punctuation heuristics |
| **Top-k retrieval only** | Simple, fast; works well for focused queries | May miss relevant context if embedding doesn't rank it highly; no re-ranking stage |
| **[Evidence N] enforced in prompt** | Enables grounding inspection and audit | LLM doesn't guarantee compliance; some claims may lack citations or hallucinate evidence indices |

**Mitigation**: Grounding score is a heuristic metric; human review of drafts is required before deployment.

---

### LLM & Cost

| Choice | Benefit | Cost |
|---|---|---|
| **Groq API (key pool + fallback)** | Resilient to rate limits & temporary outages; supports multiple models | Calls are counted against account quota; key rotation doesn't bypass rate limits |
| **Regex fallback for field extraction** | System works without API key for document ingestion | Accuracy lower on edge cases (e.g., non-standard case number formats) |
| **Semantic pattern extraction (LLM-based)** | Patterns are reusable, generalizable improvement hints | Requires LLM call per edit; not suitable for high-edit-volume workflows |

**Mitigation**: Pattern extraction is optional; user can skip if cost is a concern.

---

### Pattern Learning & Feedback

| Choice | Benefit | Cost |
|---|---|---|
| **Dedup patterns by description** | Avoids storing redundant improvement hints | Semantic duplicates with different wording may still be stored (no semantic similarity check) |
| **Top-N patterns injected into prompt** | Guides future drafts without requiring fine-tuning | Prompt grows linearly with patterns; limited to ~5 patterns due to token budget |
| **Usage-count ordering** | Most impactful patterns are prioritized | No decay over time; old patterns retain high priority indefinitely |

**Mitigation**: Users can manually delete outdated patterns via the API.

---

### Processing & Latency

| Choice | Benefit | Cost |
|---|---|---|
| **Background document processing** | Non-blocking upload; users get immediate feedback | Clients must poll `/status` endpoint; processing delay can be 10–60s depending on document size |
| **Per-page OCR fallback at 300dpi** | Handles scanned PDFs well | OCR is slow (~2–3s per page); high-res PDFs may take minutes to process |
| **No caching of retrieval results** | Simplifies state management; always fresh results | Repeated queries for the same document re-embed the query; adds latency for interactive use |

**Mitigation**: Add Redis-based retrieval caching in future versions if latency becomes a bottleneck.

---

### Frontend & UX

| Choice | Benefit | Cost |
|---|---|---|
| **Vanilla JavaScript (no framework)** | Minimal dependencies; single HTML file deployable | No component abstraction; large inline script; harder to maintain as complexity grows |
| **Dark theme UI** | Reduces eye strain; modern aesthetic | Less contrast for some accessibility standards; may need light theme option |
| **Real-time status polling** | Simple implementation; works on any network | Adds client-side complexity; unnecessary polling when document is ready |

**Mitigation**: Consider React/Vue migration if interactivity complexity exceeds 500 lines of JS.

---

### Security & Compliance

| Choice | Benefit | Cost |
|---|---|---|
| **CORS: `allow_origins=["*"]`** | Development-friendly; easy cross-domain testing | Exposes API to all origins; requires authentication layer for production |
| **File upload to disk** | Direct access to documents in data/ directory | No virus scanning; no content validation beyond file extension |
| **Groq API keys in .env** | Convenient local development | Keys can leak if .env is committed or logged; no key rotation strategy |

**Mitigation**: Require environment-based secret management (e.g., HashiCorp Vault, AWS Secrets Manager) for production.

---

### Testing & Validation

| Choice | Benefit | Cost |
|---|---|---|
| **Pytest + no external service mocks** | Real ChromaDB and SQLite in tests; higher confidence | Test suite slower; some tests require writable HOME directory for ChromaDB model cache |
| **36 passing tests** | Good coverage of core paths (processor, retrieval, improvement loop) | No integration tests with real Groq API; frontend not tested; Docker deployment untested |

**Mitigation**: Add integration test suite with recorded Groq responses; add E2E tests with Playwright.

---

## Performance Tradeoffs Summary

| Metric | Current | Scalability Path |
|---|---|---|
| **Max documents** | ~1000 | → PostgreSQL + object store |
| **Concurrent users** | 1–10 | → Kubernetes + load balancing |
| **Retrieval latency** | 100–300ms | → Redis caching + Weaviate |
| **OCR throughput** | 1 page/2–3s | → Distributed OCR workers |
| **LLM call cost** | ~$0.01–0.05/draft | → Fine-tuned local model |

---

## Conclusion

The system prioritizes **simplicity and developer productivity** over scalability. It is well-suited for:
- ✅ Prototyping and MVP deployment
- ✅ Learning systems and small teams (<100 users)
- ✅ Moderate document volumes (<10K documents)
- ✅ Low-latency local deployments

For production at scale (>10K documents, >100 concurrent users), plan to:
1. Migrate SQLite → PostgreSQL
2. Replace ChromaDB → Weaviate/Pinecone
3. Containerize with Kubernetes
4. Add Redis caching layer
5. Implement API authentication and rate limiting
6. Add comprehensive monitoring and logging
