import pytest

from predictor import RuBertPredictor

@pytest.fixture
def predictor():
    return RuBertPredictor()

def test_predictor_is_heading(predictor):
    assert (predictor.predict("Введение") == "heading")

def test_predictor_not_heading(predictor):
    assert (predictor.predict("В данной работе рассматривается проблема классификации структурных элементов.") != "heading")

def test_predictor_is_paragraph(predictor):
    assert (predictor.predict("В данной работе рассматривается проблема автоматической классификации структурных элементов научных текстов с использованием методов машинного обучения.") == "paragraph")

def test_predictor_not_paragraph(predictor):
    assert (predictor.predict("2.1 Методология исследования") != "paragraph")

def test_predictor_is_caption(predictor):
    assert (predictor.predict("Таблица 2. Сравнение результатов классификации") == "caption")

def test_predictor_not_caption(predictor):
    assert (predictor.predict("2.1 Методология исследования") != "caption")