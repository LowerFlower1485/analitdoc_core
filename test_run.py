"""
Тестовый запуск полного пайплайна обработки автомобильных документов.

Запуск (из корня проекта C:\\AnalitDoc\\scan_intellect_ai):
    .\\venv\\Scripts\\python.exe test_run.py
"""
import os
import sys
import json

# Корректный вывод кириллицы в консоли Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.pipeline import DocumentPipeline
from core.rag_engine import LocalRAGEngine

# Все пути относительны корня проекта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
SAMPLE_IMAGE = os.path.join(BASE_DIR, "test_data", "sample.jpg")


def main():
    print("=" * 60)
    print("On-Premises AI Core: обработка автомобильных документов")
    print("=" * 60)

    if not os.path.exists(SAMPLE_IMAGE):
        print(f"Ошибка: тестовое изображение не найдено: {SAMPLE_IMAGE}")
        sys.exit(1)

    # --- 1. Полный пайплайн (OCR -> VLM/LLM -> RAG) ---------------------
    print(f"\n[1] Запуск пайплайна для файла: {SAMPLE_IMAGE}")
    pipeline = DocumentPipeline(storage_dir=STORAGE_DIR)
    result = pipeline.process_document(SAMPLE_IMAGE, save_to_index=True)

    print("\n--- Результат обработки (JSON) ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if result.get("status") != "success":
        print("\nОбработка завершилась с ошибкой. См. выше.")
        sys.exit(1)

    # --- 2. Семантический поиск через RAG ------------------------------
    print("\n[2] Проверка семантического поиска (LocalRAGEngine)")
    rag = LocalRAGEngine(storage_dir=STORAGE_DIR)

    queries = [
        "Автомобиль марки LADA",
        "VIN номер",
        "Собственник Иванов",
    ]
    for q in queries:
        print(f"\n  Запрос: '{q}'")
        hits = rag.search(q, top_k=3)
        if not hits:
            print("    -> совпадений не найдено")
            continue
        for i, doc in enumerate(hits):
            fields = doc.get("extracted_data") or {}
            print(
                f"    [{i + 1}] doc_id={doc.get('document_id')} match={doc.get('match_type')} "
                f"score={doc.get('score', 0):.4f} | vin={fields.get('vin')} "
                f"| model={fields.get('vehicle_model')} | owner={fields.get('owner_name')}"
            )

    print("\n" + "=" * 60)
    print("Верификация завершена успешно.")
    print("=" * 60)


if __name__ == "__main__":
    main()
