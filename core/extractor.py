"""
Извлечение структурированных полей из результатов OCR с помощью локальной
LLM (Ollama) и строгой JSON-схемы Pydantic v2, с расчётом confidence_score.
Полностью автономно (on-premises), без внешних API.
"""
import json
import logging
import re
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator

try:
    import ollama
except ImportError:  # pragma: no cover
    ollama = None

logger = logging.getLogger("FieldExtractor")

# Кириллические символы-двойники латиницы, типичные для ошибок OCR в VIN
_VIN_TRANSLIT = str.maketrans({
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M",
    "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "У": "Y", "Х": "X", "а": "A", "в": "B", "е": "E",
    "к": "K", "м": "M", "н": "H", "о": "O", "р": "P",
    "с": "C", "т": "T", "у": "Y", "х": "X",
})


class VehicleDocumentSchema(BaseModel):
    """Схема извлекаемых полей автомобильного документа (ПТС/ЭПТС/СТС)."""
    document_type: Literal["ПТС", "ЭПТС", "СТС", "Неизвестно"] = Field(
        default="Неизвестно", description="Тип документа: ПТС, ЭПТС, СТС или Неизвестно"
    )
    vin: Optional[str] = Field(default=None, description="VIN номер (17 символов, латиница и цифры)")
    plate_number: Optional[str] = Field(default=None, description="Государственный регистрационный знак (госномер)")
    vehicle_model: Optional[str] = Field(default=None, description="Марка и модель транспортного средства")
    manufacture_year: Optional[int] = Field(default=None, description="Год выпуска транспортного средства")
    engine_number: Optional[str] = Field(default=None, description="Номер двигателя")
    chassis_number: Optional[str] = Field(default=None, description="Номер шасси (рамы)")
    color: Optional[str] = Field(default=None, description="Цвет кузова")
    owner_name: Optional[str] = Field(default=None, description="ФИО собственника или наименование юрлица")
    issue_date: Optional[str] = Field(default=None, description="Дата выдачи документа")

    @field_validator("document_type", mode="before")
    @classmethod
    def _normalize_document_type(cls, v):
        if v is None:
            return "Неизвестно"
        s = str(v).strip().upper()
        if "ЭПТС" in s or "ЭЛЕКТРОН" in s:
            return "ЭПТС"
        if "СВИДЕТЕЛЬСТВО" in s or "СТС" in s or "STS" in s:
            return "СТС"
        if "ПТС" in s or "PTS" in s or "ПАСПОРТ" in s:
            return "ПТС"
        return "Неизвестно"

    @field_validator("vin", mode="before")
    @classmethod
    def _normalize_vin(cls, v):
        return normalize_vin(v)


def normalize_vin(value: Optional[str]) -> Optional[str]:
    """Исправляет типичные опечатки OCR в VIN: кириллица -> латиница, upper, без пробелов."""
    if not value:
        return None
    normalized = value.translate(_VIN_TRANSLIT)
    normalized = normalized.upper().replace(" ", "").replace("-", "")
    return normalized or None


def _looks_like_vin(token: str) -> bool:
    """Проверка, что токен похож на VIN (не на номер шасси/кузова/бланка)."""
    if len(token) != 17:
        return False
    if not token.isalnum():
        return False
    if any(c in token for c in "IOQ"):  # запрещённые в ISO 3779 символы
        return False
    return any(c.isalpha() for c in token) and any(c.isdigit() for c in token)


def _valid_vin(value: Optional[str]) -> bool:
    """Валидация VIN: нормализация + проверка структуры (17 символов, без I/O/Q)."""
    return bool(value) and _looks_like_vin(normalize_vin(value) or "")


def find_vin_in_lines(ocr_lines: List[Any]) -> Optional[str]:
    """
    Детерминированный поиск VIN (17 символов) в строках OCR.
    Принимает список dict ({"text": ...}) или список строк.

    Защита от ложных срабатываний:
      * длина ровно 17 символов;
      * только буквы/цифры, без запрещённых символов I, O, Q;
      * обязательно есть и буквы, и цифры;
      * приоритет строки с маркером VIN; строки шасси/кузова/двигателя/бланка
        отбраковываются.
    """
    candidates = []
    for line in ocr_lines:
        text = line.get("text", "") if isinstance(line, dict) else str(line)
        for token in re.split(r"[^A-Za-zА-Яа-я0-9]+", text):
            norm = normalize_vin(token)
            if norm and _looks_like_vin(norm):
                candidates.append((norm, text.upper()))

    if not candidates:
        return None

    for norm, text in candidates:
        if re.search(r"VIN|ИДЕНТИФИКАЦИОН", text):
            return norm

    for norm, text in candidates:
        if re.search(r"ШАССИ|КУЗОВ|ДВИГАТЕЛ|РАМА|БЛАНК|СЕРИЯ", text):
            continue
        return norm

    return None


def compute_confidence_score(ocr_lines: List[Dict[str, Any]], fields: Dict[str, Any]) -> float:
    """
    Оценка уверенности итогового извлечения: 50% — средний confidence OCR,
    50% — доля успешно прошедших валидацию ключевых полей.
    """
    avg_ocr = (
        sum(l.get("confidence", 0.0) for l in ocr_lines) / len(ocr_lines)
        if ocr_lines else 0.0
    )
    checks = {
        "vin": _valid_vin(fields.get("vin")),
        "plate_number": bool(fields.get("plate_number")),
        "vehicle_model": bool(fields.get("vehicle_model")),
        "manufacture_year": isinstance(fields.get("manufacture_year"), int),
        "owner_name": bool(fields.get("owner_name")),
    }
    ratio = sum(checks.values()) / len(checks)
    score = 0.5 * avg_ocr + 0.5 * ratio
    return round(min(max(score, 0.0), 1.0), 3)


# ---------------------------------------------------------------------------
# Быстрый детерминированный парсер выписок ЭПТС (fast-path, без LLM)
# ---------------------------------------------------------------------------
def _normalize_key(s: str) -> str:
    return re.sub(r"[^A-Za-zА-Яа-я0-9]", "", str(s).upper())


# Нормализованные подписи полей -> имя поля схемы
_EPTS_LABELS = {
    "vin": ["ИДЕНТИФИКАЦИОННЫЙНОМЕР", "VIN"],
    "plate_number": ["РЕГИСТРАЦИОННЫЙЗНАК", "ГОСУДАРСТВЕННЫЙРЕГИСТРАЦИОННЫЙЗНАК"],
    "brand": ["МАРКА", "МАРКАМОДЕЛЬ"],
    "vehicle_model": ["КОММЕРЧЕСКОЕНАИМЕНОВАНИЕ", "МОДЕЛЬ"],
    "manufacture_year": ["ГОДИЗГОТОВЛЕНИЯ", "ГОДВЫПУСКА"],
    "engine_number": ["НОМЕРДВИГАТЕЛЯ"],
    "chassis_number": ["НОМЕРШАССИ", "НОМЕРРАМЫ"],
    "color": ["ЦВЕТ"],
    "owner_name": ["СОБСТВЕННИК"],
    "issue_date": ["ДАТАВЫДАЧИ", "ДАТАОФОРМЛЕНИЯ"],
}


def _match_label(token: str) -> Optional[str]:
    """Возвращает имя поля, если токен является подписью (label) в ЭПТС."""
    norm = _normalize_key(token)
    if not norm:
        return None
    for field, keys in _EPTS_LABELS.items():
        for key in keys:
            # точное совпадение или подпись начинается с ключа (например '...номер (VIN)')
            if norm == key or norm.startswith(key):
                return field
    return None


def _clean_value(v: str) -> Optional[str]:
    v = (v or "").strip()
    if not v:
        return None
    if _normalize_key(v) in ("ОТСУТСТВУЕТ", "НЕУСТАНОВЛЕН", "НЕТДАННЫХ"):
        return None
    return v


def is_epts_vypiska(lines: List[str]) -> bool:
    """Определяет, является ли документ выпиской из ЭПТС."""
    text = " ".join(lines).upper()
    return "ЭЛЕКТРОННОГО ПАСПОРТА" in text or ("ВЫПИСКА" in text and "ЭПТС" in text)


def parse_epts_fastpath(grouped_lines: List[str]) -> Dict[str, Any]:
    """
    Детерминированное извлечение полей выписки ЭПТС из пар 'Ключ | Значение'.
    Работает за доли миллисекунды без вызова LLM.
    """
    data: Dict[str, Any] = {
        "document_type": "ЭПТС",
        "vin": None, "plate_number": None, "vehicle_model": None,
        "manufacture_year": None, "engine_number": None, "chassis_number": None,
        "color": None, "owner_name": None, "issue_date": None,
    }
    brand = None
    model = None

    for line in grouped_lines:
        tokens = [t.strip() for t in line.split("|")]
        for i, tok in enumerate(tokens):
            field = _match_label(tok)
            if field is None:
                continue
            # значение — следующий токен, если он сам не является подписью
            val = None
            if i + 1 < len(tokens) and _match_label(tokens[i + 1]) is None:
                val = _clean_value(tokens[i + 1])

            if field == "brand":
                brand = val or brand
            elif field == "vehicle_model":
                model = val or model
            elif data.get(field) in (None, ""):
                data[field] = val

    # Итоговая модель: коммерческое наименование приоритетнее марки
    data["vehicle_model"] = model or brand

    # Нормализация
    data["vin"] = normalize_vin(data.get("vin"))
    if data.get("manufacture_year") is not None:
        m = re.search(r"\d{4}", str(data["manufacture_year"]))
        data["manufacture_year"] = int(m.group()) if m else None

    return data


class FieldExtractor:
    def __init__(self, model_name: str = "qwen2.5:7b"):
        self.model_name = model_name
        self._ensure_model()

    def _ensure_model(self):
        """Автоматически выбирает доступную модель qwen2.5 из ollama.list()."""
        if ollama is None:
            logger.warning("Библиотека ollama не установлена. Извлечение будет работать в режиме заглушки.")
            return
        try:
            available = self._parse_model_list(ollama.list())
            logger.info(f"Доступные модели Ollama: {available}")
            if self.model_name in available:
                return
            qwen_variants = [m for m in available if "qwen2.5" in m.lower() or "qwen2" in m.lower()]
            if qwen_variants:
                self.model_name = qwen_variants[0]
                logger.info(f"Выбрана доступная модель qwen: {self.model_name}")
            elif available:
                self.model_name = available[0]
                logger.info(f"Выбрана первая доступная модель Ollama: {self.model_name}")
        except Exception as e:
            logger.warning(f"Не удалось получить список моделей Ollama ({e}). Используется: {self.model_name}")

    @staticmethod
    def _parse_model_list(response) -> List[str]:
        if response is None:
            return []
        if hasattr(response, "models"):
            return [getattr(m, "model", str(m)) for m in response.models]
        if isinstance(response, dict) and "models" in response:
            return [m.get("model", "") if isinstance(m, dict) else getattr(m, "model", "") for m in response["models"]]
        if isinstance(response, list):
            return [m.get("model", "") if isinstance(m, dict) else getattr(m, "model", "") for m in response]
        return []

    @staticmethod
    def _system_prompt() -> str:
        return (
            "Ты — эксперт по извлечению данных из автомобильных документов РФ (ПТС, ЭПТС, СТС, "
            "включая выписки из ЭПТС).\n"
            "Проанализируй распознанный OCR-текст и заполни структуру строго по JSON-схеме.\n"
            "Правила:\n"
            "1. В VIN-номере исправляй типичные опечатки OCR: заменяй кириллические символы-двойники "
            "(А, В, Е, К, М, Н, О, Р, С, Т, У, Х) на латинские эквиваленты.\n"
            "2. VIN должен содержать ровно 17 символов (латиница и цифры, без I, O, Q).\n"
            "3. Если поле отсутствует на скане — строго ставь null. Ничего не выдумывай.\n"
            "4. owner_name заполняй ТОЛЬКО из явного поля 'Собственник'/'Владелец' (ФИО или организация).\n"
            "   Строго запрещено записывать в owner_name:\n"
            "   - организацию-изготовителя из блока 'Паспорт оформил'/'Организация-изготовитель';\n"
            "   - данные из штампа электронной подписи (ЭЦП): 'Система Электронных Паспортов', "
            "'Сертификат', 'Владелец: Электронных', 'Электронных Паспортов' и т.п.\n"
            "   Если явного блока 'Собственник' нет — ставь null.\n"
            "5. issue_date — это дата выдачи/оформления документа. Дата 'формирования выписки' "
            "не является датой выдачи.\n"
            "6. В выписке из ЭПТС обычно НЕТ собственника и госномера — ставь для них null.\n"
            "7. Соответствие полей подписям в документе:\n"
            "   - document_type: 'ПТС' (Паспорт ТС), 'ЭПТС' (электронный ПТС/выписка из ЭПТС), "
            "'СТС' (заголовок 'Свидетельство о регистрации ТС') или 'Неизвестно';\n"
            "   - 'Регистрационный знак' -> plate_number;\n"
            "   - 'Идентификационный номер (VIN)' -> vin;\n"
            "   - 'Марка' + 'Коммерческое наименование' -> vehicle_model;\n"
            "   - 'Год изготовления/выпуска' -> manufacture_year (число);\n"
            "   - 'Номер двигателя' -> engine_number ('отсутствует' -> null);\n"
            "   - 'Номер шасси (рамы)' -> chassis_number ('отсутствует' -> null);\n"
            "   - 'Цвет' -> color;\n"
            "   - 'Собственник' -> owner_name;\n"
            "   - 'Дата выдачи/оформления' -> issue_date."
        )

    def extract_fields(
        self,
        ocr_lines: List[Dict[str, Any]],
        grouped_lines: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Извлекает структурированные поля. Порядок:
        1) fast-path: выписка ЭПТС — детерминированный парсер без LLM;
        2) локальная LLM (Ollama, keep_alive);
        3) детерминированный поиск VIN в OCR.
        """
        context = grouped_lines if grouped_lines is not None else [l.get("text", "") for l in ocr_lines]
        raw_text = "\n".join(context)

        # 1) Fast-path для выписок ЭПТС — без LLM, за доли миллисекунды
        if is_epts_vypiska(context):
            data = parse_epts_fastpath(context)
            logger.info("Fast-path: выписка ЭПТС извлечена детерминированным парсером (без LLM).")
            return self._postprocess(data, context)

        # 2) Локальная LLM (Ollama)
        if ollama is None:
            data = self._dummy_fields()
        else:
            data = self._extract_local(raw_text)

        return self._postprocess(data, context)

    def _postprocess(self, data: Dict[str, Any], context: List[str]) -> Dict[str, Any]:
        """Детерминированный fallback VIN из OCR + очистка chassis_number."""
        if not _valid_vin(data.get("vin")):
            recovered = find_vin_in_lines(context)
            if recovered:
                data["vin"] = recovered
                logger.info(f"VIN восстановлен из OCR: {recovered}")
        if data.get("vin") and data.get("chassis_number") == data.get("vin"):
            data["chassis_number"] = None
        return data

    def _extract_local(self, raw_text: str) -> Dict[str, Any]:
        schema_json = VehicleDocumentSchema.model_json_schema()
        prompt = f"Распознанный текст документа:\n{raw_text}\n\nИзвлеки структурированные данные:"
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                format=schema_json,
                options={"temperature": 0.0},
                keep_alive="30m",
            )
            parsed = json.loads(response.message.content)
            return VehicleDocumentSchema(**parsed).model_dump()
        except Exception as e:
            logger.error(f"Ошибка при извлечении полей через Ollama: {e}")
            return VehicleDocumentSchema().model_dump()

    @staticmethod
    def _dummy_fields() -> Dict[str, Any]:
        return VehicleDocumentSchema(
            document_type="СТС",
            vin="XTA217030G1234567",
            plate_number="А123ВС777",
            vehicle_model="LADA PRIORA",
            manufacture_year=2021,
            owner_name="Иванов Иван Иванович",
        ).model_dump()
