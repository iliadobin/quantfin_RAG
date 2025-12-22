# Быстрый старт: Сбор корпуса

## Шаг 1: Установка зависимостей

```bash
cd /Users/dobin/workspace/NLP/qa-assistant

# Создать виртуальное окружение (если ещё не создано)
python3 -m venv venv
source venv/bin/activate

# Установить зависимости
pip install --upgrade pip
pip install -r requirements.txt
```

## Шаг 2: Проверка структуры

```bash
python scripts/test_structure.py
```

Должно вывести `✓ ALL TESTS PASSED`.

## Шаг 3: Сбор корпуса

```bash
python scripts/collect_corpus.py
```

Что происходит:
1. **Поиск arXiv**: скрипт соединяется с arXiv API и ищет 15 подходящих PDF по критериям из `configs/arxiv_search_config.yaml`
2. **Создание allowlist**: сохраняет найденные статьи в `configs/arxiv_allowlist.yaml`
3. **Скачивание anchors**: пытается скачать RiskMetrics и BCBS/Basel PDF
   - ⚠️ **Если автоматическая загрузка якорей не работает** (часто бывает):
     - Скачайте их вручную по ссылкам из `configs/anchors.yaml`
     - Поместите в `data/pdf/` как `riskmetrics_1996.pdf` и `bcbs_frtb_2019.pdf`
     - Перезапустите скрипт
4. **Скачивание arXiv PDF**: скачивает PDF из allowlist
5. **Подсчёт метаданных**: для каждого PDF:
   - считает SHA256 checksum
   - считает количество страниц
   - записывает размер файла
6. **Контроль ≤900 страниц**: если суммарно больше 900 страниц, отбрасывает лишние документы
7. **Создание манифеста**: сохраняет итоговый `configs/corpus_public.yaml`

## Проверка результата

```bash
# Проверить скачанные PDF
ls -lh data/pdf/

# Посмотреть манифест
cat configs/corpus_public.yaml | head -100

# Проверить суммарное количество страниц
grep 'total_pages:' configs/corpus_public.yaml
```

Ожидаемый результат:
- **17–20 PDF** в `data/pdf/` (2 якоря + 15 arXiv)
- **600–900 страниц** суммарно
- **Файл `configs/corpus_public.yaml`** с метаданными всех документов

## Что делать, если что-то пошло не так

### arXiv API не отвечает / медленный
- Увеличьте `time.sleep()` между запросами в `arxiv_collector.py`
- Или запустите скрипт повторно (он пропустит уже скачанные файлы)

### Anchor PDF не скачиваются
- Это нормально (часто публичные PDF переезжают)
- Найдите альтернативные публичные PDF по market risk/VaR
- Или скачайте вручную и положите в `data/pdf/`

### Слишком мало/много страниц
- Отредактируйте `configs/arxiv_search_config.yaml`:
  - измените `max_results` в `search_groups`
  - или измените `max_total_pages`
- Перезапустите `collect_corpus.py`

### Хочу другие arXiv статьи
- Отредактируйте `configs/arxiv_search_config.yaml`:
  - добавьте/уберите ключевые слова в `search_groups`
  - измените квоты `max_results`
- Удалите `configs/arxiv_allowlist.yaml`
- Перезапустите `collect_corpus.py`

## Следующие шаги

После успешного сбора корпуса:
1. Парсинг PDF → текст + чанкинг (2 стратегии): `python scripts/parse_corpus.py`
2. Chunking + построение индексов (TODO: `scripts/build_indices.py`)
3. Запуск ассистента (TODO)

