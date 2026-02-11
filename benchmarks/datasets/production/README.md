# Production datasets (DS1–DS5)

В этом проекте “production datasets” — это датасеты, в которых:

- `doc_id` **совпадает** с реальными документами из `configs/corpus_public.yaml` (например, `arxiv_1406_2133v1`, `riskmetrics_1996`).
- для DS2 qrels используются **реальные** `chunk_id` из `data/indices/public/<strategy>/chunks.jsonl`.
- для DS1/DS4/DS5 `gold_citations` ссылаются на реальные `doc_id` и `page_span` (и желательно содержат реальный `quote`).

## Быстрый старт

1) Создать шаблоны (пустые датасеты):

```bash
python scripts/create_production_dataset_templates.py
```

2) Заполнить примеры вручную / полуавтоматически.

3) Провалидировать:

```bash
python -c "from benchmarks.datasets.loader import load_dataset; from benchmarks.datasets.validator import DatasetValidator; \
ds=load_dataset('benchmarks/datasets/production/ds1_factual_qa_public.json','ds1'); \
print(DatasetValidator(strict=False).validate_ds1(ds).get_summary())"
```

4) Запустить бенчмарк на production‑датасетах:

```bash
python scripts/run_benchmark.py --datasets-dir benchmarks/datasets/production --config configs/benchmarks/all_except_v3.yaml
```

## Как наполнять датасеты (коротко)

### DS1 (Factual QA + citations)
- Выбираешь “проверяемый” вопрос (определение/формула/свойство).
- Находишь в `chunks.jsonl` фрагмент, который содержит ответ.
- Пишешь `gold_answer` (коротко и проверяемо) + `gold_citations` (doc_id + страницы + quote).

### DS2 (RetrievalQrels)
- Для каждого запроса сохраняешь список релевантных `chunk_id` (relevance=2/1).
- Это лучше размечать по **top-N выдаче** (BM25/dense/hybrid), вручную проставляя релевантность.

### DS3 (Unanswerable / traps)
- Пишешь вопрос, на который корпус **точно** не отвечает (market data, инвестиционные советы, противоречивые предпосылки).
- Указываешь `expected_behavior` (refuse/clarify/flag_uncertainty).

### DS4 (MultiHop)
- Делишь на hops (подвопросы), требующие 2+ источника/раздела.
- Даёшь `gold_answer` и citations, которые покрывают ключевые утверждения.

### DS5 (StructuredExtraction)
- Фиксируешь `output_schema` (JSON schema) и `gold_output`.
- Просишь извлечь только то, что есть в контексте; если не найдено — `null`.

## Важно про метрики

- DS2 метрики (Recall@k/nDCG/MRR/MAP) работают только если `chunk_id` в qrels совпадает с `chunks.jsonl`.
- DS1/DS4/DS5 citation precision/recall требуют реальных `doc_id` и адекватного `page_span`.


