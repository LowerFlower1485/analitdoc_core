import os
import time
import uuid
import hashlib
import logging
from typing import Dict, Any, Optional

from .ocr_engine import OCREngine
from .extractor import FieldExtractor, compute_confidence_score
from .rag_engine import LocalRAGEngine

logger = logging.getLogger("DocumentPipeline")


class DocumentPipeline:
    def __init__(
        self,
        ocr_engine: Optional[OCREngine] = None,
        field_extractor: Optional[FieldExtractor] = None,
        rag_engine: Optional[LocalRAGEngine] = None,
        storage_dir: str = "storage",
    ):
        t0 = time.perf_counter()
        self.ocr_engine = ocr_engine or OCREngine()
        t_ocr = time.perf_counter()
        self.field_extractor = field_extractor or FieldExtractor()
        t_ext = time.perf_counter()
        self.rag_engine = rag_engine or LocalRAGEngine(storage_dir=storage_dir)
        t_rag = time.perf_counter()

        self.init_timings = {
            "ocr_engine_init_sec": round(t_ocr - t0, 3),
            "field_extractor_init_sec": round(t_ext - t_ocr, 3),
            "rag_engine_init_sec": round(t_rag - t_ext, 3),
            "total_init_sec": round(t_rag - t0, 3),
        }

    @staticmethod
    def _file_hash(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def process_document(self, image_path: str, save_to_index: bool = True) -> Dict[str, Any]:
        """
        Оркестратор: предобработка+OCR -> группировка строк -> извлечение полей
        -> RAG-индексация. Возвращает результат с замерами по этапам и confidence_score.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл документа не найден: {image_path}")

        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        file_hash = self._file_hash(image_path)
        logger.info(f"Начало обработки документа {doc_id} из файла {image_path}")

        timings: Dict[str, float] = {}

        try:
            # 1. OCR
            t0 = time.perf_counter()
            ocr_lines = self.ocr_engine.extract_lines(image_path)
            grouped_lines = self.ocr_engine.group_lines(ocr_lines)
            t1 = time.perf_counter()
            timings["ocr_sec"] = round(t1 - t0, 3)

            # 2. Извлечение полей (по сгруппированным строкам)
            extracted_fields = self.field_extractor.extract_fields(ocr_lines, grouped_lines=grouped_lines)
            t2 = time.perf_counter()
            timings["extraction_sec"] = round(t2 - t1, 3)

            # 3. Индексация в RAG
            if save_to_index:
                self.rag_engine.add_document(
                    doc_id=doc_id,
                    extracted_data=extracted_fields,
                    raw_lines=ocr_lines,
                    file_hash=file_hash,
                )
            t3 = time.perf_counter()
            timings["indexing_sec"] = round(t3 - t2, 3)
            timings["total_sec"] = round(t3 - t0, 3)

            confidence = compute_confidence_score(ocr_lines, extracted_fields)

            logger.info(f"Документ {doc_id} обработан за {timings['total_sec']} сек "
                        f"(confidence={confidence}).")

            return {
                "status": "success",
                "document_id": doc_id,
                "file": os.path.basename(image_path),
                "file_hash": file_hash,
                "fields": extracted_fields,
                "confidence_score": confidence,
                "ocr_lines_count": len(ocr_lines),
                "ocr_lines": ocr_lines,
                "grouped_lines": grouped_lines,
                "processing_time_sec": timings["total_sec"],
                "timings": timings,
            }

        except Exception as e:
            timings["total_sec"] = round(time.perf_counter() - t0, 3)
            logger.error(f"Ошибка обработки документа {image_path}: {e}")
            return {
                "status": "error",
                "file": os.path.basename(image_path),
                "error_message": str(e),
                "processing_time_sec": timings["total_sec"],
                "timings": timings,
            }
