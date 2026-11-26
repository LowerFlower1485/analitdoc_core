# AnalitDoc Core

Автономное (on-premises) вычислительное ядро системы интеллектуальной обработки
автомобильных документов (ПТС, ЭПТС, СТС): OCR, извлечение структурированных полей
и семантический поиск по архиву.

## Стек

| Этап | Технология |
|------|-----------|
| OCR | PaddleOCR 3.7 (`engine="onnxruntime"`) + OpenCV-предобработка |
| Экстракция полей | Ollama (`qwen2.5`) + детерминированный fast-path парсер, Pydantic v2 |
| Хранение / поиск | SQLite + Ollama (`bge-m3`), гибридный поиск (SQL + векторы) |

Полностью локально, без внешних API.

## Модули (`core/`)

- `ocr_engine.py` — `OCREngine`: предобработка (ресайз, CLAHE, подавление синего канала), OCR, группировка строк.
- `extractor.py` — `FieldExtractor` + `VehicleDocumentSchema`: извлечение полей (VIN, госномер, модель, собственник…), `confidence_score`.
- `rag_engine.py` — `LocalRAGEngine`: SQLite-хранилище, дедупликация по VIN/хэшу, гибридный поиск.
- `pipeline.py` — `DocumentPipeline`: оркестратор `process_document()`.

## Запуск

```powershell
.\venv\Scripts\python.exe test_run.py      # smoke-тест на test_data/sample.jpg
.\venv\Scripts\python.exe stress_test.py   # бенчмарк и стресс-тест
```

## Требования

- Python 3.11
- Пакеты: `paddleocr`, `onnxruntime`, `opencv-python`, `pydantic` v2, `ollama`, `numpy`
- Модели Ollama: `qwen2.5:3b` (экстракция), `bge-m3` (эмбеддинги)
- Модели PaddleOCR: `PP-OCRv5_mobile_det`, `eslav_PP-OCRv5_mobile_rec`
