# Сбор корпуса PDF: Реализация и использование

## Что реализовано

### 1. Полная структура проекта
```
qa-assistant/
├── configs/                    # Конфигурации
│   ├── anchors.yaml           # Якорные документы (RiskMetrics, BCBS)
│   └── arxiv_search_config.yaml  # Параметры поиска arXiv
├── ingest/sources/            # Модули сбора PDF
│   ├── arxiv_collector.py     # Сбор allowlist через arXiv API
│   └── pdf_downloader.py      # Загрузка + метаданные + sha256
├── knowledge/models.py        # Pydantic модели (Document, CorpusManifest)
├── scripts/
│   ├── collect_corpus.py      # Главный скрипт сбора корпуса
│   └── test_structure.py      # Проверка структуры и зависимостей
├── requirements.txt           # Все зависимости
├── README.md                  # Полное описание проекта
└── QUICKSTART.md             # Пошаговая инструкция
```

### 2. Функциональность сбора корпуса

#### ArxivCollector (`ingest/sources/arxiv_collector.py`)
- Подключается к arXiv API
- Ищет статьи по 5 тематическим группам:
  - risk-neutral/martingale pricing (4 статьи)
  - Black-Scholes/Greeks (3 статьи)
  - PDE/finite difference (3 статьи)
  - Monte Carlo/variance reduction (3 статьи)
  - American options/LSMC (2 статьи)
- Фильтры:
  - только категории `q-fin.PR`, `q-fin.MF`, `q-fin.RM`
  - приоритет tutorial/lecture/survey (в title/abstract)
  - исключает crypto/blockchain/HFT
  - длина: 15–100 страниц
- Создаёт `configs/arxiv_allowlist.yaml` с метаданными

#### PDFDownloader (`ingest/sources/pdf_downloader.py`)
- Скачивает anchor документы (RiskMetrics + BCBS)
- Скачивает arXiv PDF из allowlist
- Для каждого PDF:
  - вычисляет SHA256 checksum
  - считает количество страниц (PyMuPDF)
  - сохраняет размер файла
  - записывает timestamp
- Контроль ≤900 страниц: автоматически отбрасывает лишние документы
- Создаёт `configs/corpus_public.yaml` — итоговый манифест

### 3. Итоговые файлы

После запуска `python scripts/collect_corpus.py`:

- **`configs/arxiv_allowlist.yaml`**: список найденных arXiv статей
- **`configs/corpus_public.yaml`**: финальный манифест корпуса
  - метаданные всех документов
  - sha256 checksums
  - количество страниц
  - пути к PDF
  - timestamps
- **`data/pdf/`**: папка с скачанными PDF (2 якоря + ~15 arXiv)

### 4. Особенности реализации

#### Воспроизводимость
- Все конфигурации в YAML
- arXiv allowlist хранит точные arxiv_id
- Checksums позволяют проверить идентичность файлов
- Timestamps записывают время сбора

#### Экономия токенов
- Embeddings: локально (intfloat/e5-small-v2)
- Парсинг PDF: локально (PyMuPDF)
- Только LLM API: DeepSeek для генерации/judge

#### Контроль качества корпуса
- Узкий домен: pricing + немного risk
- Приоритет tutorial/lecture/survey
- Фильтрация нерелевантных тем
- Автоматический контроль ≤900 страниц

## Как использовать

### Минимальный путь (3 команды)

```bash
cd /Users/dobin/workspace/NLP/qa-assistant

# 1. Установить зависимости
pip install -r requirements.txt

# 2. Проверить структуру
python scripts/test_structure.py

# 3. Собрать корпус
python scripts/collect_corpus.py
```

### Что делать с результатом

После сбора корпуса у вас есть:
- **Все PDF в `data/pdf/`**
- **Манифест `configs/corpus_public.yaml`** со всеми метаданными

Следующие шаги (TODO в основном проекте):
1. Парсинг PDF → текст + page spans
2. Chunking (2 стратегии)
3. Построение индексов (BM25 + vector)
4. RAG пайплайны v1–v5
5. Генерация датасетов DS1–DS5

## Настройка под свои нужды

### Изменить количество arXiv статей

Отредактируйте `configs/arxiv_search_config.yaml`:

```yaml
search_groups:
  risk_neutral_martingale:
    max_results: 4  # изменить здесь
```

### Изменить лимит страниц

В `configs/arxiv_search_config.yaml`:

```yaml
filters:
  max_total_pages: 700  # изменить (оставить ~200 для якорей)
```

Или в `scripts/collect_corpus.py`:

```python
manifest = downloader.build_corpus_manifest(
    all_docs,
    max_pages=900,  # изменить здесь
    ...
)
```

### Добавить другие якорные документы

Отредактируйте `configs/anchors.yaml`:

```yaml
anchors:
  - id: my_document
    title: "My Document Title"
    source_type: industry
    url: "https://..."
    license: "..."
    estimated_pages: 100
    description: "..."
```

### Изменить ключевые слова поиска

В `configs/arxiv_search_config.yaml` → `search_groups` → `keywords`.

## Troubleshooting

### arXiv API медленный
- Норма: arXiv API может быть медленным
- Решение: просто подождать или перезапустить (пропустит уже скачанные)

### Anchor PDF не скачиваются
- Частая проблема: публичные URL меняются
- Решение:
  1. Найти PDF вручную
  2. Скачать в `data/pdf/` как `riskmetrics_1996.pdf` и `bcbs_frtb_2019.pdf`
  3. Перезапустить скрипт

### Недостаточно страниц
- Увеличить `max_results` в search groups
- Или добавить новые search groups

### Слишком много страниц
- Уменьшить `max_results` в search groups
- Или уменьшить `max_total_pages` в filters

## Итого: что получено

✅ Полный pipeline сбора корпуса:
- Автоматический поиск arXiv через API
- Загрузка PDF с метаданными
- Контроль качества и объёма
- Воспроизводимость

✅ Готовый корпус:
- 2 якоря (risk/регуляторика)
- ~15 arXiv (pricing)
- ≤900 страниц суммарно
- Все метаданные в YAML

✅ Документация:
- README.md — общее описание проекта
- QUICKSTART.md — пошаговая инструкция
- Комментарии в коде

Следующий шаг: парсинг PDF и построение индексов (будет реализовано далее).

