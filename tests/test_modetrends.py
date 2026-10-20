"""Tests hors réseau avec réponses synthétiques, distinctes des résultats du modèle."""
import base64
import io
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modetrends import (APIError, HFClient, ROOT, confusion_matrix, cost_scenarios,
                        create_masks, dataset_pairs, make_splits, metrics, prepare_image,
                        visualize)


def encoded(array):
    data = io.BytesIO()
    Image.fromarray(np.array(array, dtype=np.uint8)).save(data, format="PNG")
    return base64.b64encode(data.getvalue()).decode()


def response(code=200, content=None, headers=None):
    result = Mock(status_code=code, headers=headers or {})
    result.json.return_value = content
    return result


def test_dataset_50_pairs():
    assert len(dataset_pairs()) == 50


def test_disjoint_reproducible_splits():
    a = make_splits(dataset_pairs())
    assert a == make_splits(dataset_pairs())
    assert [len(a[k]) for k in ["train", "validation", "test"]] == [30, 10, 10]
    assert len(set(sum(a.values(), []))) == 50


def test_perfect_metrics_absent_classes_excluded():
    truth = np.array([[0, 4], [4, 6]], dtype=np.uint8)
    result = metrics(confusion_matrix(truth, truth))
    assert result["miou"] == 1
    assert result["macro_dice"] == 1
    assert result["classes"][7]["iou"] is None


def test_known_imperfect_metrics():
    truth = np.array([[4, 4], [6, 6]])
    pred = np.array([[4, 6], [6, 6]])
    result = metrics(confusion_matrix(truth, pred))
    assert result["classes"][4]["iou"] == .5
    assert result["classes"][6]["iou"] == pytest.approx(2 / 3)
    assert result["miou"] == pytest.approx(7 / 12)


def test_false_positive_class_counts():
    result = metrics(confusion_matrix(np.array([[0]]), np.array([[7]])))
    assert result["classes"][7]["iou"] == 0
    assert result["miou"] == 0


def test_ignored_pixels():
    result = confusion_matrix(np.array([[255, 4]]), np.array([[7, 4]]))
    assert result.sum() == 1
    assert metrics(result)["miou"] == 1


def test_empty_metrics_not_fabricated():
    assert metrics(np.zeros((18, 18)))["miou"] is None


def test_mask_background_does_not_erase_foreground():
    segments = [{"label": "Upper-clothes", "score": .9, "mask": encoded([[255, 0]])},
                {"label": "Background", "score": 1, "mask": encoded([[255, 255]])}]
    assert create_masks(segments, 2, 1).tolist() == [[4, 0]]


def test_overlap_uses_score_and_nearest_resize():
    segments = [{"label": "Dress", "score": .8, "mask": encoded([[255]])},
                {"label": "Pants", "score": .2, "mask": encoded([[255]])}]
    assert (create_masks(segments, 4, 4) == 7).all()


@pytest.mark.parametrize("segment", [
    {"label": "Unknown", "mask": ""},
    {"label": "Dress", "mask": "not base64"},
    {"label": "Dress", "mask": encoded([[255]]), "score": float("nan")}])
def test_bad_api_masks_rejected(segment):
    with pytest.raises(APIError):
        create_masks([segment], 1, 1)


def test_prepare_image_valid_png(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGBA", (200, 100)).save(image)
    payload, size = prepare_image(image, 100)
    assert size == (100, 50)
    assert Image.open(io.BytesIO(payload)).mode == "RGB"
    # Vérifier aussi le chemin complet de rendu sans appeler le service.
    truth = np.full((100, 200), 4, dtype=np.uint8)
    prediction = np.full((100, 200), 6, dtype=np.uint8)
    figure = tmp_path / "synthetic_visualization.png"
    visualize(image, truth, prediction, figure)
    with Image.open(figure) as rendered:
        assert rendered.width > 1000 and rendered.height > 300


def test_retry_429_then_success():
    session = Mock()
    session.post.side_effect = [response(429, headers={"Retry-After": "2"}),
                               response(200, [{"label": "Background"}])]
    sleep = Mock()
    result, _, attempts = HFClient("test", session=session, sleep=sleep).segment(b"image")
    assert attempts == 2
    sleep.assert_called_once_with(2)
    assert result[0]["label"] == "Background"


@pytest.mark.parametrize("code", [401, 402, 403, 404, 410])
def test_permanent_error_no_retry(code):
    session = Mock()
    session.post.return_value = response(code)
    with pytest.raises(APIError, match=str(code)):
        HFClient("test", session=session).segment(b"image")
    assert session.post.call_count == 1


def test_timeouts_bounded():
    session = Mock()
    session.post.side_effect = requests.Timeout()
    with pytest.raises(APIError):
        HFClient("test", session=session, sleep=Mock()).segment(b"image")
    assert session.post.call_count == 3


def test_invalid_json_rejected():
    session = Mock()
    session.post.return_value = response(200)
    session.post.return_value.json.side_effect = ValueError()
    with pytest.raises(APIError, match="JSON"):
        HFClient("test", session=session).segment(b"image")


def test_cost_formula():
    scenario = cost_scenarios()
    assert scenario["estimated_usd"] == pytest.approx(145.8333333)
    assert scenario["dedicated_always_on_usd"] == 360
    assert scenario["requests_per_second"] == pytest.approx(.19290123)


def test_no_secret_in_notebook():
    for path in ROOT.glob("*.ipynb"):
        nb = json.loads(path.read_text("utf-8"))
        assert "hf_" not in json.dumps(nb)
