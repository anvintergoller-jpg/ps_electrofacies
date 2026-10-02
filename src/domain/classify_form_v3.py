# -*- coding: utf-8 -*-
"""
Классификация формы аномалии ПС (v3.0).

См. docs/CLASSIFICATION_RULES_v3.md, §7.

На вход — результат split_interval_into_elements (dict с элементами
кровля / плато / подошва). На выходе — тип формы, уверенность
и короткое объяснение.

Отличие от v2.x:
    • v2.x работала по трём опорным точкам и порогам между ними;
    • v3.0 работает по структурному описанию элементов.

7 форм + unknown-shape:
    bell              — Кр(Н) | – | Пд(М)
    funnel            — Кр(М) | – | Пд(Н)
    v-shape           — Кр(Н) | – | Пд(Н)
    trapezoid-top     — Кр(Н) | + | Пд(М)
    trapezoid-bottom  — Кр(М) | + | Пд(Н)
    trapezoid-middle  — Кр(Н) | + | Пд(Н)
    cylinder          — Кр(М) | + | Пд(М)
    unknown-shape     — не подходит ни под одну структуру

Формула confidence — стартовая, калибруется после пилота на 5
скважинах. Не откалиброванная вероятность, а эвристическая оценка
«насколько выражена форма».

Файл v3.0. Работает параллельно с classify_form.py (v2.x).
Старый код не трогаем.
"""

from __future__ import annotations


# Версия правил v3.0.
RULES_VERSION_V3 = "3.0"


# ===========================================================================
#  Вспомогательное
# ===========================================================================

def _symmetry_top_bot(top_length, bot_length, form_type):
    """
    Модификатор симметрии кровли и подошвы (см. v3.0 §8.3).

    Осмыслен только для форм с обеими плавными линиями:
    v-shape и trapezoid-middle. Для остальных — n/a.

    ratio = top_length / bot_length
      • ratio ∈ [0.5, 2.0] → symmetric
      • ratio > 2.0        → asymmetric-top
      • ratio < 0.5        → asymmetric-bot
      • иначе              → n/a (нет обеих плавных линий, или
                             bot_length ≈ 0)
    """
    if form_type not in ("v-shape", "trapezoid-middle"):
        return "n/a"

    if bot_length is None or bot_length < 1e-6:
        return "n/a"

    ratio = top_length / bot_length
    if ratio > 2.0:
        return "asymmetric-top"
    if ratio < 0.5:
        return "asymmetric-bot"
    return "symmetric"


# ===========================================================================
#  Уверенность (confidence) — стартовые формулы
# ===========================================================================
#
# Все формулы — предварительные. Смысл: чем более «выражена» форма
# (длиннее плато, чётче положение максимума), тем выше уверенность.
#
# После пилота на 5 скважинах эти формулы можно пересмотреть.
# Они не откалиброваны, это эвристика для сортировки и фильтрации,
# не для принятия решений.

def _conf_cylinder(plateau_length_frac):
    """Чем длиннее плато, тем увереннее cylinder."""
    if plateau_length_frac is None:
        return 0.5
    return min(1.0, plateau_length_frac * 1.2)


def _conf_trapezoid(plateau_length_frac):
    """
    Трапеции: чем длиннее плато, тем увереннее.
    Стартово от 0.7 (плато уже найдено) до 1.0.
    """
    if plateau_length_frac is None:
        return 0.7
    return min(1.0, 0.7 + 0.3 * plateau_length_frac)


def _conf_bell(max_position):
    """bell: max aSP у подошвы. Чем ближе max_position к 1, тем увереннее."""
    if max_position is None:
        return 0.5
    return max(0.5, min(1.0, (max_position - 0.5) / 0.5))


def _conf_funnel(max_position):
    """funnel: max aSP у кровли. Чем ближе max_position к 0, тем увереннее."""
    if max_position is None:
        return 0.5
    return max(0.5, min(1.0, (0.5 - max_position) / 0.5))


def _conf_v_shape(max_position):
    """v-shape: max aSP в середине. Чем ближе к 0.5, тем увереннее."""
    if max_position is None:
        return 0.5
    return max(0.3, min(1.0, 1.0 - abs(max_position - 0.5) * 2))


# ===========================================================================
#  Главная функция классификации
# ===========================================================================

def classify_form_v3(elements, params):
    """
    Определяет форму аномалии по разбиению на элементы.

    Параметры
    ---------
    elements : dict
        Результат split_interval_into_elements (elements_v3.py).
    params : dict
        Секция classification_v3 из config.yaml.
        Используется только boundary_frac для интерпретации
        max_position при отсутствии плато.

    Возвращает
    ----------
    dict:
        form_type           — одна из 8 форм (7 + unknown-shape)
        confidence          — 0.0 .. 1.0
        reason              — короткое объяснение
        rules_version       — "3.0"
        symmetry_top_bot    — модификатор симметрии (v3.0 §8.3)
    """
    result = {
        "form_type": "unknown-shape",
        "confidence": 0.3,
        "reason": "не подходит ни под одну структуру",
        "rules_version": RULES_VERSION_V3,
        "symmetry_top_bot": "n/a",
    }

    has_plateau = elements.get("has_plateau", False)
    top_type = elements.get("top_type", "abs")
    bot_type = elements.get("bot_type", "abs")

    top_length = elements.get("top_length_m")
    bot_length = elements.get("bot_length_m")

    # ------------------------------------------------------------------
    # Есть плато — трапеции и cylinder
    # ------------------------------------------------------------------
    if has_plateau:
        plateau_frac = elements.get("plateau_length_frac")

        # Обе границы мгновенные → cylinder (прямоугольник).
        if top_type == "М" and bot_type == "М":
            result["form_type"] = "cylinder"
            result["confidence"] = _conf_cylinder(plateau_frac)
            result["reason"] = "обе границы мгновенные, длинное плато"
            return result

        # Кровля наклонная, подошва мгновенная → trapezoid-top.
        if top_type == "Н" and bot_type == "М":
            result["form_type"] = "trapezoid-top"
            result["confidence"] = _conf_trapezoid(plateau_frac)
            result["reason"] = "плавный вход, плато, резкий выход"
            result["symmetry_top_bot"] = _symmetry_top_bot(
                top_length, bot_length, "trapezoid-top"
            )
            return result

        # Кровля мгновенная, подошва наклонная → trapezoid-bottom.
        if top_type == "М" and bot_type == "Н":
            result["form_type"] = "trapezoid-bottom"
            result["confidence"] = _conf_trapezoid(plateau_frac)
            result["reason"] = "резкий вход, плато, плавный выход"
            result["symmetry_top_bot"] = _symmetry_top_bot(
                top_length, bot_length, "trapezoid-bottom"
            )
            return result

        # Обе границы наклонные → trapezoid-middle.
        if top_type == "Н" and bot_type == "Н":
            result["form_type"] = "trapezoid-middle"
            result["confidence"] = _conf_trapezoid(plateau_frac)
            result["reason"] = "обе границы плавные, плато в середине"
            result["symmetry_top_bot"] = _symmetry_top_bot(
                top_length, bot_length, "trapezoid-middle"
            )
            return result

        # Есть плато, но тип границы "abs" — сюда не должны попадать,
        # но на всякий случай.
        result["form_type"] = "unknown-shape"
        result["confidence"] = 0.3
        result["reason"] = (
            f"плато есть, но типы границ нестандартные: "
            f"Кр={top_type}, Пд={bot_type}"
        )
        return result

    # ------------------------------------------------------------------
    # Плато нет — треугольники (bell / funnel / v-shape)
    # ------------------------------------------------------------------
    max_position = elements.get("max_position")

    if max_position is None:
        result["form_type"] = "unknown-shape"
        result["confidence"] = 0.3
        result["reason"] = "нет плато и не определён максимум aSP"
        return result

    boundary_frac = params.get("boundary_frac", 0.15)

    # Максимум у кровли → funnel.
    if max_position < boundary_frac:
        result["form_type"] = "funnel"
        result["confidence"] = _conf_funnel(max_position)
        result["reason"] = "резкий вход, плавный выход (максимум у кровли)"
        return result

    # Максимум у подошвы → bell.
    if max_position > 1.0 - boundary_frac:
        result["form_type"] = "bell"
        result["confidence"] = _conf_bell(max_position)
        result["reason"] = "плавный вход, резкий выход (максимум у подошвы)"
        return result

    # Максимум в середине → v-shape.
    result["form_type"] = "v-shape"
    result["confidence"] = _conf_v_shape(max_position)
    result["reason"] = "обе границы плавные, максимум в середине"
    result["symmetry_top_bot"] = _symmetry_top_bot(
        top_length, bot_length, "v-shape"
    )
    return result


__all__ = ["RULES_VERSION_V3", "classify_form_v3"]