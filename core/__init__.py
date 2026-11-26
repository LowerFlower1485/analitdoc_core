"""Вычислительное ядро системы интеллектуальной обработки автомобильных документов."""
import sys
import logging

# Корректный вывод кириллицы в консоли Windows
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

# Единая настройка логирования для всего ядра
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from .ocr_engine import OCREngine
from .extractor import (
    FieldExtractor,
    VehicleDocumentSchema,
    normalize_vin,
    find_vin_in_lines,
    compute_confidence_score,
    parse_epts_fastpath,
    is_epts_vypiska,
)
from .rag_engine import LocalRAGEngine
from .pipeline import DocumentPipeline

__all__ = [
    "OCREngine",
    "FieldExtractor",
    "VehicleDocumentSchema",
    "normalize_vin",
    "find_vin_in_lines",
    "compute_confidence_score",
    "parse_epts_fastpath",
    "is_epts_vypiska",
    "LocalRAGEngine",
    "DocumentPipeline",
]
