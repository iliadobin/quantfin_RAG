# QA Assistant - Quantitative Finance

QA-ассистент по quantitative finance (derivatives pricing + market risk) с Telegram-интерфейсом, основанный на RAG и бенчмарках.

## Ключевые характеристики

- **Домен**: узко — деривативы и pricing/Greeks/hedging
- **Корпус**: ~600–900 страниц публичных PDF (воспроизводимо)
- **LLM**: DeepSeek API (с prompt caching для экономии)
- **Embeddings**: локально на CPU (`intfloat/e5-small-v2`)
- **RAG версии**: 5 пайплайнов (dense, hybrid+rerank, multiquery, parent-child, evidence-validation)
- **Бенчмарк**: 5 датасетов (~800 примеров): factual QA, retrieval, unanswerable/traps, multi-hop, structured extraction
- **UI**: Telegram бот

## Быстрый старт

### 1. Установка зависимостей

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 2. Настройка

Создайте `.env` файл:

```bash
cp .env.example .env
# Отредактируйте .env и добавьте DEEPSEEK_API_KEY
```

### 3. Сбор корпуса

**Шаг 1: Построить arXiv allowlist**

```bash
python scripts/collect_corpus.py
```

Скрипт:
- Соберёт 15 arXiv PDF через API (по конфигу `configs/arxiv_search_config.yaml`)
- Скачает anchor документы (RiskMetrics, BCBS/Basel) — **может потребовать ручной загрузки, если URL недоступны**
- Посчитает метаданные (sha256, page count)
- Создаст манифест `configs/corpus_public.yaml` (≤900 страниц)

**Важно**: Если anchor PDF не скачиваются автоматически, скачайте их вручную и поместите в `data/pdf/`:
- `riskmetrics_1996.pdf`
- `bcbs_frtb_2019.pdf`

Затем перезапустите скрипт.

**Шаг 2: Проверить корпус**

```bash
ls -lh data/pdf/
cat configs/corpus_public.yaml | head -50
```

### 4. Парсинг и индексация

```bash
python scripts/parse_corpus.py
python scripts/build_indices.py
```

### 5. Запуск Telegram бота

```bash
# TODO: будет реализовано
python apps/telegram_bot/bot.py
```

## Структура проекта

```
qa-assistant/
├── configs/               # Конфигурации
│   ├── anchors.yaml
│   ├── arxiv_search_config.yaml
│   ├── arxiv_allowlist.yaml  (generated)
│   └── corpus_public.yaml    (generated)
├── ingest/
│   └── sources/          # Сбор PDF
│       ├── arxiv_collector.py
│       └── pdf_downloader.py
├── knowledge/            # Модели данных
│   └── models.py
├── rag/                  # RAG пайплайны (TODO)
├── benchmarks/           # Датасеты и метрики (TODO)
├── llm/                  # DeepSeek клиент (TODO)
├── apps/telegram_bot/    # Telegram UI (TODO)
├── scripts/              # CLI утилиты
│   └── collect_corpus.py
├── data/                 # Данные (не в git)
│   ├── pdf/
│   ├── parsed/
│   └── indices/
└── requirements.txt
```

## Оптимизация расхода токенов

Проект оптимизирован под минимальный расход DeepSeek API:

- **Embeddings**: локально (0 API токенов)
- **Rerank**: локальный cross-encoder (0 API токенов)
- **Prompt caching**: static prefix + dynamic suffix → cache hit
- **Частичный LLM-judge**: только DS1/DS3/DS4/DS5, и только на нужных примерах

Ожидаемый расход (весь проект): **~50–150M токенов**.

## Состав корпуса (public)

### Якоря (risk/регуляторика)
- RiskMetrics Technical Document
- BCBS/Basel Market Risk Standard (FRTB)

### arXiv (pricing)
- 15 PDF из категорий `q-fin.PR`, `q-fin.MF`:
  - risk-neutral pricing / martingale measure
  - Black–Scholes / Greeks / hedging
  - PDE / finite difference methods
  - Monte Carlo / variance reduction
  - American options / LSMC

## Лицензия

Учебный проект. Корпус собран из публичных источников с указанием лицензий в манифесте.

## TODO

- [ ] Парсинг PDF → текст + page spans
- [ ] Чанкинг (2 стратегии)
- [ ] Построение индексов (BM25 + vector)
- [ ] RAG пайплайны v1–v5
- [ ] Генерация датасетов DS1–DS5
- [ ] Benchmark runner
- [ ] Telegram бот
- [ ] Тесты

