"""
Стресс-тест, профилирование и сравнительный бенчмарк (до/после оптимизации).

Запуск (из корня проекта C:\\AnalitDoc\\scan_intellect_ai):
    .\\venv\\Scripts\\python.exe stress_test.py
"""
import os
import sys
import json
import shutil

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from core.pipeline import DocumentPipeline
from core.rag_engine import LocalRAGEngine
from core.extractor import find_vin_in_lines, _looks_like_vin

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_SCAN = os.path.join(BASE_DIR, "test_data", "raw_epts.jpeg")
STRESS_STORAGE = os.path.join(BASE_DIR, "storage_stress")
DEDUP_STORAGE = os.path.join(BASE_DIR, "storage_dedup")

SEP = "=" * 72

# Базовые замеры ДО этой оптимизации (mobile_det + paddle engine + LLM)
BEFORE = {
    "init_sec": 1.44,
    "ocr_sec": 51.85,
    "extraction_sec": 26.91,
    "indexing_sec": 0.0,
    "total_sec": 78.75,
}


def section(title: str):
    print(f"\n{SEP}\n{title}\n{SEP}")


# ---------------------------------------------------------------------------
# 1. Прогон на реальном скане (после оптимизации)
# ---------------------------------------------------------------------------
def profile_real_scan():
    section("[1] Прогон на реальном скане raw_epts.jpeg (после оптимизации)")

    if not os.path.exists(REAL_SCAN):
        print(f"Ошибка: файл не найден {REAL_SCAN}")
        sys.exit(1)

    pipeline = DocumentPipeline(storage_dir=STRESS_STORAGE)
    print("\n--- Время инициализации (сек) ---")
    print(json.dumps(pipeline.init_timings, ensure_ascii=False, indent=2))

    result = pipeline.process_document(REAL_SCAN, save_to_index=False)
    if result.get("status") != "success":
        print(f"\nОшибка: {result}")
        sys.exit(1)

    print("\n--- Извлечённые поля (JSON) ---")
    print(json.dumps(result["fields"], ensure_ascii=False, indent=2))
    print(f"confidence_score = {result['confidence_score']}")

    print("\n--- Сгруппированные строки (первые 20, табличная структура) ---")
    for i, line in enumerate(result["grouped_lines"][:20]):
        print(f"  [{i:2d}] {line!r}")

    print("\n--- Замеры по этапам (сек, после) ---")
    print(json.dumps(result["timings"], ensure_ascii=False, indent=2))

    return result, pipeline.init_timings


# ---------------------------------------------------------------------------
# 2. Сравнительный отчёт до/после
# ---------------------------------------------------------------------------
def compare_report(after: dict, init_timings: dict):
    section("[2] Сравнительный отчёт: до vs после оптимизации")
    t = after.get("timings", {})
    rows = [
        ("Инициализация", BEFORE["init_sec"], init_timings.get("total_init_sec", 0)),
        ("OCR", BEFORE["ocr_sec"], t.get("ocr_sec", 0)),
        ("LLM-экстракция", BEFORE["extraction_sec"], t.get("extraction_sec", 0)),
        ("RAG-индексация", BEFORE["indexing_sec"], t.get("indexing_sec", 0)),
        ("ИТОГО", BEFORE["total_sec"], t.get("total_sec", 0)),
    ]
    print(f"{'Этап':<18} {'До (сек)':>10} {'После (сек)':>12} {'Ускорение':>10}")
    print("-" * 52)
    for name, b, a in rows:
        speedup = f"{b / a:.2f}x" if a and a > 0 else "n/a"
        print(f"{name:<18} {b:>10.2f} {a:>12.2f} {speedup:>10}")
    print("-" * 52)
    print("Примечание: до — PaddleOCR (paddle engine) + LLM; после — PaddleOCR (onnxruntime) + fast-path парсер ЭПТС.")


# ---------------------------------------------------------------------------
# 3. Проверка fallback-поиска VIN
# ---------------------------------------------------------------------------
def test_vin_fallback():
    section("[3] Проверка fallback-поиска VIN")

    cases = [
        ("Валидный VIN на строке с маркером VIN",
         [{"text": "Идентификационный номер (VIN) XTA217030G0123456"}], "XTA217030G0123456"),
        ("VIN отдельной строкой без маркера", [{"text": "XTA217030G0123456"}], "XTA217030G0123456"),
        ("Кириллица-двойник в VIN -> латиница", [{"text": "ХТА217030G0123456"}], "XTA217030G0123456"),
        ("Запрещённый символ I в VIN", [{"text": "XTA217030I0123456"}], None),
        ("Запрещённый символ O в VIN", [{"text": "XTA21703O00123456"}], None),
        ("Чисто числовой 17-значный номер шасси", [{"text": "12345678901234567"}], None),
        ("Номер шасси на строке 'Шасси (рама)'", [{"text": "Шасси (рама) XTA217030G0123456"}], None),
        ("Номер кузова на строке 'Кузов'", [{"text": "Кузов XTA217030G0123456"}], None),
        ("Номер бланка (короткий)", [{"text": "Бланк серия 77 ТК 123456"}], None),
        ("Приоритет строки VIN над Шасси",
         [{"text": "Шасси XTA217030G0123456"}, {"text": "VIN Z8T4C1FS0N1234567"}], "Z8T4C1FS0N1234567"),
        ("Сгруппированная строка 'Номер кузова | VIN-значение'",
         [{"text": "Номер кузова (кабины, прицепа) | X96C41R13LEH00006"}], None),
    ]

    passed = 0
    for desc, lines, expected in cases:
        got = find_vin_in_lines(lines)
        ok = got == expected
        passed += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")
        if not ok:
            print(f"        ожидалось={expected!r}  получено={got!r}")

    unit_cases = [
        ("XTA217030G0123456", True), ("12345678901234567", False),
        ("XTA217030I0123456", False), ("XTA217030Q0123456", False),
        ("XTA217030O0123456", False), ("XTA217030", False),
    ]
    for token, expected in unit_cases:
        got = _looks_like_vin(token)
        ok = got == expected
        passed += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] _looks_like_vin({token!r}) -> {got}")

    print(f"\nИтог VIN-тестов: {passed}/{len(cases) + len(unit_cases)} пройдено")
    return passed == (len(cases) + len(unit_cases))


# ---------------------------------------------------------------------------
# 4. Защита от дубликатов (SQLite)
# ---------------------------------------------------------------------------
def test_rag_dedup():
    section("[4] Защита от дубликатов в LocalRAGEngine (SQLite, по VIN/хэшу файла)")

    if os.path.exists(DEDUP_STORAGE):
        shutil.rmtree(DEDUP_STORAGE)
    rag = LocalRAGEngine(storage_dir=DEDUP_STORAGE)

    vin = "XTA217030G0123456"
    line = [{"text": "VIN XTA217030G0123456 LADA PRIORA 2021"}]

    try:
        rag.add_document("doc_first", {"vin": vin, "vehicle_model": "LADA PRIORA"}, line)
        n1 = len(rag)
        print(f"  После 1-го добавления: документов = {n1}")

        rag.add_document("doc_second", {"vin": vin, "vehicle_model": "LADA VESTA"}, line)
        n2 = len(rag)
        print(f"  После 2-го добавления (тот же VIN): документов = {n2}")

        rag.add_document("doc_third", {"vin": vin, "vehicle_model": "LADA VESTA"}, line, file_hash="abc123")
        n3 = len(rag)
        print(f"  После 3-го добавления (тот же VIN + file_hash): документов = {n3}")

        all_docs = rag.get_all()
        model = all_docs[0]["extracted_data"]["vehicle_model"] if all_docs else None
        doc_id = all_docs[0]["document_id"] if all_docs else None
        print(f"  Итоговая запись: doc_id={doc_id}, model={model}")

        ok = (n1 == 1) and (n2 == 1) and (n3 == 1) and (model == "LADA VESTA")
        print(f"  [{'PASS' if ok else 'FAIL'}] дубликат не создан, запись обновлена")
        return ok
    finally:
        rag.close()


# ---------------------------------------------------------------------------
def main():
    print(SEP)
    print("Стресс-тест, профилирование и сравнительный бенчмарк")
    print(SEP)

    result, init_timings = profile_real_scan()
    compare_report(result, init_timings)

    results = {
        "VIN fallback": test_vin_fallback(),
        "RAG дедупликация": test_rag_dedup(),
    }

    section("СВОДКА")
    all_ok = all(results.values())
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{'ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ' if all_ok else 'ЕСТЬ ПРОВАЛЫ'}")


if __name__ == "__main__":
    main()
