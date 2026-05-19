# Coursework Reviewer

Инструмент для **автоматического форматирования курсовых работ** в формате `.docx`. Нейросеть на основе RuBERT классифицирует каждый абзац документа (`paragraph` / `heading` / `caption`), после чего применяет к нему правила оформления из YAML-конфига.

---

## Содержание

- [Как это работает](#как-это-работает)
- [Структура проекта](#структура-проекта)
- [Требования](#требования)
- [Установка](#установка)
- [Быстрый старт — веб-интерфейс](#быстрый-старт--веб-интерфейс)
- [Командная строка](#командная-строка)
- [Конфигурация форматирования](#конфигурация-форматирования)
- [Обучение модели](#обучение-модели)
- [Оценка модели](#оценка-модели)
- [Тесты](#тесты)
- [Бенчмарк скорости](#бенчмарк-скорости)
- [Качество модели](#качество-модели)

---

## Как это работает

```
.docx файл
    │
    ▼
Извлечение абзацев (python-docx)
    │
    ▼
RuBERT классификатор
(cointegrated/rubert-tiny2, дообученный)
    │
    ├── paragraph  ──► отступ 1.25 см, шрифт 12pt, justify
    ├── heading    ──► уровень 1/2/3, жирный, Times New Roman
    └── caption    ──► курсив 11pt, по центру
    │
    ▼
Применение стилей из format_config.yaml
    │
    ▼
Сохранённый .docx
```

Уровень заголовка (1–3) определяется автоматически: сначала из стиля Word (`Heading N` / `Заголовок N`), при его отсутствии — по нумерации в начале текста (`1.2.3 → уровень 3`).

**Автоматически пропускаются:**
- **Титульный лист** — все абзацы до первого разрыва страницы (явный `page break` или разрыв секции)
- **Список литературы** — все абзацы начиная с заголовка, совпадающего с шаблоном: «Список литературы», «Список использованных источников», «Литература», «References» и т.д.

---

## Структура проекта

```
coursework-reviewer/
├── app.py                  # Streamlit веб-интерфейс
├── format_docx.py          # Логика форматирования + CLI
├── format_config.yaml      # Правила оформления по типам элементов
├── bench.py                # Бенчмарк скорости инференса
├── requirements.txt        # Зависимости
│
├── src/
│   ├── predictor.py        # Класс RuBertPredictor (инференс)
│   ├── train.py            # Обучение модели
│   ├── eval.py             # Оценка на тестовой выборке
│   ├── dataset.py          # PyTorch Dataset для обучения
│   └── tests/
│       └── test_predictor.py  # Unit-тесты предсказателя
│
├── models/
│   └── best_model/         # Веса дообученной модели (не в git)
│
├── data/
│   ├── raw/                # Исходные .docx файлы
│   └── processed/          # Датасет в формате JSON (train/val/test)
│
└── notebooks/
    └── dataset.ipynb       # Подготовка и анализ датасета
```

---

## Требования

- Python 3.10+
- CUDA (опционально, для ускорения инференса и обучения)

Основные зависимости:

| Пакет | Назначение |
|---|---|
| `torch` | Фреймворк для инференса и обучения |
| `transformers` | RuBERT модель и токенайзер |
| `python-docx` | Чтение и запись `.docx` |
| `streamlit` | Веб-интерфейс |
| `PyYAML` | Парсинг конфига форматирования |
| `datasets` | Загрузка датасета при обучении |
| `scikit-learn` | Метрики при обучении/оценке |

---

## Установка

```bash
# 1. Клонировать репозиторий
git clone <repo-url>
cd coursework-reviewer

# 2. Создать виртуальное окружение
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# 3. Установить зависимости
pip install -r requirements.txt

# Если нужна поддержка CUDA (замените cu126 на вашу версию CUDA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

> **Важно:** папка `models/best_model/` с весами модели должна присутствовать в корне проекта. Либо обучите модель самостоятельно (см. [Обучение модели](#обучение-модели)), либо положите готовые веса в эту директорию.

---

## Быстрый старт — веб-интерфейс

```bash
streamlit run app.py
```

Откроется браузер по адресу `http://localhost:8501`.

**Использование:**

1. Нажмите **Browse files** и выберите `.docx` файл курсовой работы.
2. Нажмите кнопку **Форматировать** — модель загрузится и обработает документ.
3. После завершения появится кнопка **Скачать отформатированный файл**.

Прогресс отображается спиннером. Исходный файл не изменяется — вы скачиваете отдельный результат.

---

## Командная строка

Если веб-интерфейс не нужен, можно использовать `format_docx.py` напрямую:

```bash
# Базовое использование
python format_docx.py input.docx output.docx

# С явным указанием конфига и размера батча
python format_docx.py input.docx output.docx --config format_config.yaml --batch-size 32

# С другой директорией модели
python format_docx.py input.docx output.docx --model path/to/model
```

**Аргументы:**

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `input` | — | Путь к исходному `.docx` |
| `output` | — | Путь для сохранения результата |
| `--config`, `-c` | `format_config.yaml` | YAML-файл с правилами оформления |
| `--batch-size`, `-b` | `16` | Размер батча при инференсе |
| `--model` | `models/best_model` | Директория с весами модели |

Пример вывода:
```
Loading model…
Classifying 87 paragraphs (batch_size=16)…
Saved → output.docx
  headings: 12, captions: 5, paragraphs: 70, skipped: 0
```

---

## Конфигурация форматирования

Файл `format_config.yaml` задаёт стили для каждого типа элемента. Можно менять под требования конкретной кафедры или вуза.

```yaml
heading_1:
  font:
    name: Times New Roman
    size: 16          # pt
    bold: true
    italic: false
    # color: "1F3864" # hex, опционально
  paragraph:
    alignment: center   # left | center | right | justify
    space_before: 18    # pt
    space_after: 6      # pt
    line_spacing: 1.5   # множитель (1.0, 1.5, 2.0)
    first_line_indent: 0 # cm

heading_2:
  ...

heading_3:
  ...

paragraph:
  font:
    name: Times New Roman
    size: 12
    bold: false
    italic: false
  paragraph:
    alignment: justify
    line_spacing: 1.5
    first_line_indent: 1.25  # красная строка

caption:
  font:
    name: Times New Roman
    size: 11
    bold: false
    italic: true
  paragraph:
    alignment: center
    line_spacing: 1.0
```

Поддерживаемые ключи `paragraph`: `alignment`, `space_before`, `space_after`, `line_spacing`, `first_line_indent`.  
Поддерживаемые ключи `font`: `name`, `size`, `bold`, `italic`, `color` (HEX строка).

---

## Обучение модели

Обучение выполняется в директории `src/`. Перед запуском убедитесь, что датасет подготовлен в `data/processed/` в формате Hugging Face JSON (файлы `train.jsonl`, `validation.jsonl`, `test.jsonl`).

Каждая запись датасета должна иметь поля:
```json
{"text": "...", "label_id": 0, "is_negative": false}
```
Метки: `0 = paragraph`, `1 = heading`, `2 = caption`.

```bash
cd src
python train.py
```

Скрипт:
- Загружает базовую модель `cointegrated/rubert-tiny2`
- Обучает 3 эпохи с AdamW (lr=1e-5) и линейным прогревом (10%)
- Использует взвешенный CrossEntropy для балансировки классов
- Сохраняет лучшую по macro F1 модель в `models/best_model/`
- Выводит F1 по каждому классу и матрицу ошибок после каждой эпохи

Обучение на GPU занимает несколько минут.

---

## Оценка модели

```bash
cd src
python eval.py
```

Вычисляет F1-macro, F1 по каждому классу и матрицу ошибок на тестовой выборке из `data/processed/`.

---

## Тесты

```bash
cd src
pytest tests/
```

Тесты проверяют корректность предсказаний модели на типичных примерах:

| Тест | Входной текст | Ожидаемый класс |
|---|---|---|
| `test_predictor_is_heading` | `"Введение"` | `heading` |
| `test_predictor_is_paragraph` | Длинное предложение из текста | `paragraph` |
| `test_predictor_is_caption` | `"Таблица 2. Сравнение результатов"` | `caption` |

---

## Бенчмарк скорости

```bash
python bench.py your_document.docx

# Тестировать конкретные размеры батча
python bench.py your_document.docx --batch-sizes 1,8,16,32,64
```

Пример вывода:
```
Document: your_document.docx
Paragraphs: 87

Loading model…
  sequential (predict one-by-one): 12.40s  (142.5 ms/para)
  batch_size=8   :  1.20s  (13.8 ms/para)
  batch_size=16  :  0.85s  (9.8 ms/para)
  batch_size=32  :  0.80s  (9.2 ms/para)
  batch_size=64  :  0.81s  (9.3 ms/para)
```

Батчинг даёт ~15x ускорение по сравнению с последовательным предсказанием.

---

## Качество модели

Результаты на тестовой выборке (дообученная `rubert-tiny2`):

| Класс | F1-score |
|---|---|
| paragraph | 0.99 |
| heading | 0.98 |
| caption | 0.99 |
| **macro avg** | **0.99** |
