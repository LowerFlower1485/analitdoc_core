"""
OCR-движок на базе PaddleOCR.

Совместим с PaddleOCR 3.x (установлена 3.7.0). Реализует извлечение текстовых
строк, уверенности и координат bounding_box [x1, y1, x2, y2], а также
предобработку изображений (ресайз, CLAHE, подавление синего канала) и
группировку OCR-блоков по строкам для сохранения табличной структуры.
"""
import os
import logging
from typing import List, Dict, Any, Optional

# Отключаем проверку доступности внешних хостов моделей (строго локальная работа)
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from paddleocr import PaddleOCR
except ImportError:  # pragma: no cover
    PaddleOCR = None

logger = logging.getLogger("OCREngine")

# Предустановленные модели PaddleOCR 3.x (быстрая мобильная детекция + русское распознавание)
_DEFAULT_DET = "PP-OCRv5_mobile_det"
_DEFAULT_REC = "eslav_PP-OCRv5_mobile_rec"
_SERVER_DET = "PP-OCRv5_server_det"


class OCREngine:
    def __init__(
        self,
        lang: str = "ru",
        use_angle_cls: bool = False,
        det_model: str = "mobile",  # "mobile" (быстро) или "server" (точно)
        max_side: int = 1800,
        blue_suppression: float = 0.6,
    ):
        """
        :param lang: язык распознавания.
        :param use_angle_cls: определение поворота текстовых строк (медленнее на ~1–2 сек).
        :param det_model: модель детекции — "mobile" (быстро) или "server" (точно).
        :param max_side: максимум длинной стороны изображения (px) после предобработки.
        :param blue_suppression: коэффициент подавления синего канала (0.0–1.0).
        """
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.det_model = det_model
        self.max_side = max_side
        self.blue_suppression = blue_suppression
        self.ocr = None
        self.ocr_version = self._detect_version()
        self._init_ocr()

    @staticmethod
    def _detect_version() -> int:
        if PaddleOCR is None:
            return 0
        return 3 if hasattr(PaddleOCR, "predict") else 2

    def _init_ocr(self):
        if PaddleOCR is None:
            logger.warning("PaddleOCR не установлен. Будет использоваться фиктивный OCR-движок.")
            return

        det_name = _SERVER_DET if self.det_model == "server" else _DEFAULT_DET
        try:
            if self.ocr_version >= 3:
                # engine="onnxruntime" — быстрый ONNX Runtime вместо Paddle Inference,
                # обходит баг oneDNN/PIR и использует те же веса PaddleOCR (русская модель).
                kwargs = dict(
                    text_detection_model_name=det_name,
                    text_recognition_model_name=_DEFAULT_REC,
                    use_textline_orientation=self.use_angle_cls,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    engine="onnxruntime",
                )
                self.ocr = PaddleOCR(**kwargs)
            else:  # pragma: no cover — запасной путь для PaddleOCR 2.x
                self.ocr = PaddleOCR(use_angle_cls=self.use_angle_cls, lang=self.lang, show_log=False)
            logger.info(f"PaddleOCR инициализирован (детекция: {det_name}, engine: onnxruntime).")
        except Exception as e:
            logger.error(f"Ошибка инициализации PaddleOCR (onnxruntime): {e}; пробую paddle engine...")
            try:
                if self.ocr_version >= 3:
                    self.ocr = PaddleOCR(
                        text_detection_model_name=det_name,
                        text_recognition_model_name=_DEFAULT_REC,
                        use_textline_orientation=self.use_angle_cls,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        enable_mkldnn=False,
                    )
                else:  # pragma: no cover
                    self.ocr = PaddleOCR(use_angle_cls=self.use_angle_cls, lang=self.lang, show_log=False)
                logger.info("PaddleOCR инициализирован (paddle engine, fallback).")
            except Exception as e2:
                logger.error(f"Ошибка инициализации PaddleOCR (paddle): {e2}")
                self.ocr = None

    # ------------------------------------------------------------- preprocess
    def _preprocess(self, img: Any) -> Any:
        """Ресайз, подавление синего канала и выравнивание контраста (CLAHE)."""
        if img is None:
            return None
        h, w = img.shape[:2]
        long_side = max(h, w)
        if long_side > self.max_side:
            scale = self.max_side / long_side
            img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                             interpolation=cv2.INTER_AREA)

        # Подавление синего канала (очистка от синих печатей/штампов)
        b, g, r = cv2.split(img)
        b = cv2.multiply(b, self.blue_suppression)
        img = cv2.merge([b, g, r])

        # Мягкое выравнивание контраста (CLAHE по L-каналу LAB)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, bb = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, bb]), cv2.COLOR_LAB2BGR)
        return img

    # --------------------------------------------------------------- extract
    def extract_lines(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Извлекает текстовые строки, уверенность и координаты рамок с изображения.

        Возвращает список словарей:
            [{"text": str, "confidence": float, "bounding_box": [x1, y1, x2, y2]}, ...]
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл изображения не найден: {image_path}")

        if self.ocr is None:
            return self._fallback_lines()

        try:
            if self.ocr_version >= 3:
                return self._extract_v3(image_path)
            return self._extract_v2(image_path)  # pragma: no cover
        except Exception as e:
            logger.error(f"Ошибка при выполнении OCR: {e}")
            return []

    def _extract_v3(self, image_path: str) -> List[Dict[str, Any]]:
        input_data = image_path
        if cv2 is not None:
            img = cv2.imread(image_path)
            if img is not None:
                input_data = self._preprocess(img)

        results = []
        pred = self.ocr.predict(input=input_data)
        if not pred:
            return results

        page = pred[0]
        texts = page.get("rec_texts", []) or []
        scores = page.get("rec_scores", []) or []
        boxes = page.get("rec_boxes", None)
        polys = page.get("rec_polys", []) or []

        for i, text in enumerate(texts):
            box = self._to_bbox(boxes, polys, i)
            conf = float(scores[i]) if i < len(scores) else 0.0
            results.append({
                "text": str(text).strip(),
                "confidence": round(conf, 3),
                "bounding_box": box,
            })
        return results

    @staticmethod
    def _to_bbox(boxes, polys, idx) -> List[int]:
        if boxes is not None:
            try:
                row = boxes[idx]
                if len(row) == 4:
                    return [int(float(v)) for v in row]
            except (IndexError, TypeError):
                pass
        if idx < len(polys):
            poly = polys[idx]
            xs = [float(p[0]) for p in poly]
            ys = [float(p[1]) for p in poly]
            return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
        return [0, 0, 0, 0]

    def _extract_v2(self, image_path: str) -> List[Dict[str, Any]]:  # pragma: no cover
        results = []
        ocr_result = self.ocr.ocr(image_path, cls=self.use_angle_cls)
        if not ocr_result:
            return results
        for line in ocr_result:
            for box_points, (text, conf) in line:
                xs = [p[0] for p in box_points]
                ys = [p[1] for p in box_points]
                results.append({
                    "text": str(text).strip(),
                    "confidence": round(float(conf), 3),
                    "bounding_box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                })
        return results

    # --------------------------------------------------------------- grouping
    @staticmethod
    def group_lines(lines: List[Dict[str, Any]], y_tolerance: Optional[float] = None) -> List[str]:
        """
        Группирует OCR-блоки по строкам на основе Y-координат bounding_box.
        Сортировка сверху вниз, внутри строки — слева направо, объединение через ' | '.
        Сохраняет табличную структуру ПТС/ЭПТС (label | value).

        Порог объединения по умолчанию — 60% медианной высоты строки (устойчиво
        к плотной/разреженной вёрстке и не «склеивает» соседние строки).
        """
        if not lines:
            return []

        ordered = sorted(lines, key=lambda l: l["bounding_box"][1])
        if y_tolerance is None:
            heights = [l["bounding_box"][3] - l["bounding_box"][1] for l in ordered]
            heights = [h for h in heights if h > 0]
            med = sorted(heights)[len(heights) // 2] if heights else 20
            y_tolerance = max(med * 0.6, 4.0)

        rows: List[List[Dict[str, Any]]] = []
        for line in ordered:
            cy = (line["bounding_box"][1] + line["bounding_box"][3]) / 2.0
            placed = False
            for row in rows:
                row_cys = [(l["bounding_box"][1] + l["bounding_box"][3]) / 2.0 for l in row]
                row_cy = sum(row_cys) / len(row_cys)
                if abs(cy - row_cy) <= y_tolerance:
                    row.append(line)
                    placed = True
                    break
            if not placed:
                rows.append([line])

        grouped: List[str] = []
        for row in sorted(rows, key=lambda r: min(l["bounding_box"][1] for l in r)):
            row.sort(key=lambda l: l["bounding_box"][0])
            text = " | ".join(l["text"].strip() for l in row if l["text"].strip())
            if text:
                grouped.append(text)
        return grouped

    # ------------------------------------------------------------- fallback
    @staticmethod
    def _fallback_lines() -> List[Dict[str, Any]]:
        return [
            {"text": "СТС Свидетельство о регистрации ТС", "confidence": 0.985,
             "bounding_box": [10, 10, 200, 50]},
            {"text": "VIN: XTA217030G1234567", "confidence": 0.972,
             "bounding_box": [10, 60, 300, 100]},
            {"text": "Госномер: А123ВС777", "confidence": 0.991,
             "bounding_box": [10, 110, 200, 150]},
        ]
