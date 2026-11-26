"""
Локальное векторное хранилище и гибридный поиск (RAG).

Хранение — SQLite (storage/documents.db): метаданные + эмбеддинги (BLOB).
Поиск гибридный: сначала точное/частичное совпадение по VIN, госномеру и
собственнику через SQL, затем семантический поиск по векторам (bge-m3).
"""
import os
import re
import json
import sqlite3
import logging
import numpy as np
from typing import List, Dict, Any, Optional

try:
    import ollama
except ImportError:  # pragma: no cover
    ollama = None

logger = logging.getLogger("LocalRAGEngine")


class LocalRAGEngine:
    def __init__(
        self,
        storage_dir: str = "storage",
        embedding_model: str = "bge-m3",
    ):
        self.storage_dir = storage_dir
        self.embedding_model = embedding_model
        self.db_path = os.path.join(storage_dir, "documents.db")
        os.makedirs(storage_dir, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                file_hash TEXT,
                vin TEXT,
                plate_number TEXT,
                owner_name TEXT,
                vehicle_model TEXT,
                manufacture_year INTEGER,
                document_type TEXT,
                extracted_data TEXT,
                search_text TEXT,
                embedding BLOB,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        for col in ("vin", "plate_number", "owner_name", "file_hash", "document_id"):
            self.conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_documents_{col} ON documents({col})"
            )
        self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def __len__(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()
        return int(row["c"])

    def get_all(self) -> List[Dict[str, Any]]:
        """Все записи хранилища (для инспекции/отладки)."""
        rows = self.conn.execute("SELECT * FROM documents ORDER BY id").fetchall()
        return [self._row_to_doc(r, match_type="stored") for r in rows]

    # ----------------------------------------------------------- embeddings
    def _get_embedding(self, text: str) -> List[float]:
        if ollama is not None:
            try:
                response = ollama.embeddings(model=self.embedding_model, prompt=text)
                embedding = self._extract_embedding(response)
                if embedding:
                    return embedding
            except Exception as e:
                logger.error(f"Ошибка получения эмбеддинга ({self.embedding_model}): {e}")
        return self._dummy_embedding(text)

    @staticmethod
    def _extract_embedding(response) -> Optional[List[float]]:
        if isinstance(response, dict):
            return response.get("embedding")
        if hasattr(response, "embedding"):
            return response.embedding
        try:
            return response["embedding"]
        except (KeyError, TypeError):
            return None

    @staticmethod
    def _dummy_embedding(text: str) -> List[float]:
        rng = np.random.default_rng(abs(hash(text)) % (2 ** 32))
        vec = rng.standard_normal(1024).astype(np.float32)
        return (vec / np.linalg.norm(vec)).tolist()

    # -------------------------------------------------------------- public
    def add_document(
        self,
        doc_id: str,
        extracted_data: Dict[str, Any],
        raw_lines: List[Dict[str, Any]],
        file_hash: Optional[str] = None,
    ):
        """
        Добавляет документ в хранилище. Защита от дубликатов: если уже есть
        запись с тем же file_hash / VIN / doc_id — она обновляется, а не дублируется.
        """
        raw_text = " ".join([line.get("text", "") for line in raw_lines])
        vin = extracted_data.get("vin")
        search_text = (
            f"Документ {doc_id}. "
            f"Тип: {extracted_data.get('document_type')}. "
            f"VIN: {vin}. "
            f"Госномер: {extracted_data.get('plate_number')}. "
            f"Модель: {extracted_data.get('vehicle_model')}. "
            f"Год: {extracted_data.get('manufacture_year')}. "
            f"Собственник: {extracted_data.get('owner_name')}. "
            f"Текст: {raw_text}"
        )

        embedding = self._get_embedding(search_text)
        blob = np.array(embedding, dtype=np.float32).tobytes()
        extracted_json = json.dumps(extracted_data, ensure_ascii=False)

        values = (
            doc_id, file_hash, vin,
            extracted_data.get("plate_number"),
            extracted_data.get("owner_name"),
            extracted_data.get("vehicle_model"),
            extracted_data.get("manufacture_year"),
            extracted_data.get("document_type"),
            extracted_json, search_text, blob,
        )

        existing_id = self._find_duplicate(vin, file_hash, doc_id)
        if existing_id is not None:
            self.conn.execute(
                """
                UPDATE documents SET
                    document_id=?, file_hash=?, vin=?, plate_number=?, owner_name=?,
                    vehicle_model=?, manufacture_year=?, document_type=?,
                    extracted_data=?, search_text=?, embedding=?, updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                values + (existing_id,),
            )
            logger.info(f"Документ {doc_id} (VIN {vin}) обновлён в RAG-хранилище (всего: {len(self)}).")
        else:
            self.conn.execute(
                """
                INSERT INTO documents
                    (document_id, file_hash, vin, plate_number, owner_name, vehicle_model,
                     manufacture_year, document_type, extracted_data, search_text, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            logger.info(f"Документ {doc_id} (VIN {vin}) добавлен в RAG-хранилище (всего: {len(self)}).")
        self.conn.commit()

    def _find_duplicate(self, vin: Optional[str], file_hash: Optional[str], doc_id: str) -> Optional[int]:
        if file_hash:
            row = self.conn.execute(
                "SELECT id FROM documents WHERE file_hash = ? LIMIT 1", (file_hash,)
            ).fetchone()
            if row:
                return row["id"]
        if vin:
            row = self.conn.execute(
                "SELECT id FROM documents WHERE vin = ? LIMIT 1", (vin,)
            ).fetchone()
            if row:
                return row["id"]
        row = self.conn.execute(
            "SELECT id FROM documents WHERE document_id = ? LIMIT 1", (doc_id,)
        ).fetchone()
        return row["id"] if row else None

    # --------------------------------------------------------------- search
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Гибридный поиск: SQL (VIN/госномер/собственник) + семантический (векторы)."""
        if not query or not query.strip():
            return []

        lexical = self._lexical_search(query, limit=top_k)
        semantic = self._semantic_search(query, top_k=top_k)

        merged: Dict[str, Dict[str, Any]] = {}
        for doc in lexical:
            merged[doc["document_id"]] = doc
        for doc in semantic:
            did = doc["document_id"]
            if did in merged:
                merged[did]["match_type"] = "lexical+semantic"
                merged[did]["score"] = max(merged[did]["score"], doc["score"])
            else:
                merged[did] = doc

        results = list(merged.values())
        # lexical-совпадения раньше, внутри — по убыванию score
        results.sort(key=lambda d: (0 if "lexical" in d["match_type"] else 1, -(d["score"] or 0.0)))
        return results[:top_k]

    def _lexical_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        tokens = [t for t in re.split(r"[^A-Za-zА-Яа-я0-9]+", query.strip()) if t]
        found: Dict[str, Dict[str, Any]] = {}
        for tok in tokens:
            like = f"%{tok}%"
            rows = self.conn.execute(
                """
                SELECT * FROM documents
                WHERE vin = ? OR vin LIKE ? OR plate_number = ? OR plate_number LIKE ?
                      OR owner_name LIKE ? OR document_id = ?
                LIMIT ?
                """,
                (tok, like, tok, like, like, tok, limit),
            ).fetchall()
            for r in rows:
                found[r["document_id"]] = self._row_to_doc(r, match_type="lexical", score=1.0)
        return list(found.values())

    def _semantic_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM documents").fetchall()
        if not rows:
            return []

        matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        q = np.asarray(self._get_embedding(query), dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sims = np.dot(matrix / norms, q)

        top_k = min(top_k, len(rows))
        order = np.argsort(sims)[::-1][:top_k]
        return [
            self._row_to_doc(rows[i], match_type="semantic", score=float(sims[i]))
            for i in order
        ]

    @staticmethod
    def _row_to_doc(row, match_type: str, score: Optional[float] = None) -> Dict[str, Any]:
        return {
            "document_id": row["document_id"],
            "file_hash": row["file_hash"],
            "extracted_data": json.loads(row["extracted_data"] or "{}"),
            "match_type": match_type,
            "score": score,
        }
