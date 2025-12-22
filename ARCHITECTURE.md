# ARCHITECTURE (current state)

Этот документ фиксирует **текущее состояние проекта** (что уже реализовано) и минимальный контекст по структуре.

## Что уже сделано

### 1) Сбор публичного корпуса PDF (≤ 900 страниц)

Реализован полностью автоматизированный пайплайн:
- **Сбор arXiv allowlist через arXiv API** (15 PDF под домен *pricing + немного risk*)
- **Загрузка PDF** (якоря + arXiv) в `data/pdf/`
- **Метаданные для каждого PDF**:
  - `sha256` checksum
  - `page_count` (PyMuPDF)
  - `file_size_bytes`
  - `retrieved_at`
- **Контроль ограничения**: итоговый корпус **не превышает 900 страниц** (лишние документы отбрасываются)
- **Итоговый манифест**: `configs/corpus_public.yaml`

Фактически сейчас проект умеет воспроизводимо собрать корпус из публичных PDF и описать его манифестом.

## Структура модулей (важное)

### Конфигурации
- `configs/anchors.yaml` — список “якорных” PDF (risk/регуляторика) с URL и описанием
- `configs/arxiv_search_config.yaml` — настройки поиска arXiv (категории, ключевые слова, квоты, фильтры)
- `configs/arxiv_allowlist.yaml` — **генерируется**: выбранные arXiv статьи (id/title/authors/pdf_url/…)
- `configs/corpus_public.yaml` — **генерируется**: финальный манифест корпуса (включая `sha256`, `page_count`)

### Код
- `ingest/sources/arxiv_collector.py` — сбор allowlist через arXiv API (по группам тем, со скорингом)
- `ingest/sources/pdf_downloader.py` — загрузка PDF, подсчёт страниц, sha256, сбор манифеста (≤900 страниц)
- `knowledge/models.py` — pydantic-модели `Document`, `CorpusManifest`

### Скрипты
- `scripts/collect_corpus.py` — “одна кнопка”: allowlist → download → manifest
- `scripts/test_structure.py` — проверка зависимостей/структуры перед запуском

## Артефакты (что появляется после запуска)

После `python scripts/collect_corpus.py`:
- `data/pdf/*.pdf` — скачанные PDF
- `configs/arxiv_allowlist.yaml` — allowlist arXiv
- `configs/corpus_public.yaml` — итоговый манифест корпуса

## Что сделано (Epic B): Ingest & Indexing

- ✅ Парсинг PDF → текст по страницам + нормализация + page spans: `scripts/parse_corpus.py`
- ✅ Chunking (2 стратегии): `fixed` (page-local, с char offsets) и `section_aware` (heading-based, с section_path)
- ✅ Индексация: BM25 + vector index (embeddings: `intfloat/e5-small-v2` на CPU): `scripts/build_indices.py`

## Что сделано (Epic C): RAG Pipelines

### Архитектура RAG системы

```
User Query
    ↓
[Retriever] → Get relevant chunks from corpus
    ↓
[Reranker] → (Optional) Reorder by relevance
    ↓
[Generator] → Generate answer with citations
    ↓
[Guardrails] → Validate evidence, detect unanswerable
    ↓
Answer + Citations
```

### Реализованные компоненты

#### 1. Контракты (`rag/contracts.py`)
- **Protocols** для всех компонентов: `Retriever`, `Reranker`, `Generator`, `Guardrail`, `Pipeline`
- Единый интерфейс для всех пайплайнов

#### 2. Модели (`knowledge/models.py`)
- **RetrievedChunk**: chunk + score + metadata
- **Citation**: doc_id + page_span + quote + score
- **Answer**: text + citations + confidence + refusal_reason + trace
- **RetrievalTrace**: полная трассировка пайплайна для debugging

#### 3. LLM Client (`llm/deepseek_client.py`)
- **DeepSeekClient**: обёртка над DeepSeek API
- **Кеширование**: SQLite-based cache для повторных запросов
- **Token tracking**: подсчёт входных/выходных токенов
- **Prompt caching**: consistent structure для cache hits
- **Retry logic**: автоматические повторы при ошибках

#### 4. Retrievers (`rag/retrievers/`)
- ✅ **DenseRetriever**: semantic search (sentence-transformers + FAISS)
- ✅ **BM25Retriever**: lexical search (rank_bm25)
- ✅ **HybridRetriever**: BM25 + dense with reciprocal rank fusion (RRF)
- ✅ **MultiQueryRetriever**: template-based query expansion + fusion

#### 5. Rerankers (`rag/rerankers/`)
- ✅ **CrossEncoderReranker**: локальный cross-encoder (no API calls)

#### 6. Generators (`rag/generators/`)
- ✅ **CitationGenerator**: генерация ответов с inline citations [1], [2]
- ✅ **Prompts**: cache-friendly промпты для DeepSeek
- Автоматический mapping citations → source chunks

#### 7. Guardrails (`rag/guardrails/`)
- ✅ **EvidenceValidator**: проверка что claims поддержаны цитатами
  - Rule-based (default, fast)
  - LLM-based (optional, costs tokens)
- ✅ **UnanswerableDetector**: обнаружение out-of-scope вопросов
  - Pre-retrieval checks (pattern matching)
  - Post-retrieval checks (low scores)

### Реализованные пайплайны (`rag/pipelines/`)

#### ✅ RAGv1: Dense (`rag_v1_dense.py`)
**Описание**: Simple baseline using dense retrieval.

**Шаги**:
1. Dense retrieval (sentence transformer + FAISS)
2. Generate answer with citations
3. Optional: unanswerable detection

**Применение**: Fast, semantic search. Good for well-defined questions.

#### ✅ RAGv2: Hybrid + Rerank (`rag_v2_hybrid.py`)
**Описание**: Combines BM25 + dense retrieval, then reranks.

**Шаги**:
1. Hybrid retrieval (BM25 + dense with RRF fusion)
2. Rerank with cross-encoder
3. Generate answer with citations
4. Optional: unanswerable detection

**Применение**: Best recall and precision. Recommended for production.

#### ✅ RAGv3: Multi-Query (`rag_v3_multiquery.py`)
**Описание**: Expands query using domain templates (cheap, no LLM).

**Шаги**:
1. Expand query using templates (pricing, Greeks, methods, etc.)
2. Retrieve for each query variant
3. Fuse results with RRF
4. Optional: reranking
5. Generate answer with citations

**Применение**: Improves coverage for multi-aspect questions. Token-efficient.

**Template strategies**:
- `pricing`: "How to price X", "Pricing formula for X"
- `greeks`: "Greeks for X", "Delta and gamma of X"
- `assumptions`: "Assumptions for X", "Requirements for X"
- `methods`: "Methods for X", "Approaches to X"

#### ✅ RAGv4: Parent-Child (`rag_v4_parent_child.py`)
**Описание**: Retrieves small chunks for precision, expands to parent context.

**Шаги**:
1. Retrieve small "child" chunks for precision
2. Expand to "parent" context (page or section)
3. Generate answer from parent context with child-level citations

**Применение**: Better context for complex topics requiring surrounding information.

#### ✅ RAGv5: Evidence Validation (`rag_v5_evidence.py`)
**Описание**: Generates answer, then validates evidence support.

**Шаги**:
1. Hybrid retrieval + reranking
2. Generate answer with citations
3. Validate evidence (rule-based or LLM)
4. Refuse or adjust confidence if validation fails

**Применение**: Maximum reliability. Ensures all claims have supporting evidence.

### Token Optimization

Все пайплайны оптимизированы для минимального расхода токенов:

1. **Prompt caching**: Consistent system prompts для DeepSeek cache hits
2. **Local models**: Embeddings и reranking используют локальные модели (no API calls)
3. **Template-based expansion**: RAGv3 uses rules instead of LLM для query expansion
4. **Rule-based guardrails**: Evidence validation использует правила by default (LLM optional)
5. **Result caching**: LLM responses cached с deterministic keys

### Примеры использования

См. `examples/rag_example.py` для полных примеров всех пайплайнов.

Быстрый пример:
```python
from rag.pipelines import RAGv2Hybrid
from rag.retrievers import DenseRetriever, BM25Retriever, HybridRetriever
from rag.rerankers import CrossEncoderReranker
from llm import DeepSeekClient

# Setup
dense = DenseRetriever(index_dir="data/indices/public/fixed", ...)
bm25 = BM25Retriever(index_dir="data/indices/public/fixed", ...)
hybrid = HybridRetriever(bm25, dense)
reranker = CrossEncoderReranker()
llm = DeepSeekClient()

# Create pipeline
pipeline = RAGv2Hybrid(hybrid, reranker, llm)

# Run
answer = pipeline.run("What is the Black-Scholes formula?")
print(answer.text)
print(f"Citations: {len(answer.citations)}")
print(f"Confidence: {answer.confidence:.2f}")
```

## Что ещё НЕ сделано (следующие этапы)

- Бенчмарк (DS1–DS5), раннер, метрики (Epic D)
- Baselines: LLM direct (Epic E)
- Telegram-бот UI (Epic F)
- Tests & Performance optimization (Epic G)


