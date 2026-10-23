from catdog.config import LABELS
from catdog.inference.predictor import Predictor, format_table
from tests.conftest import jpeg_bytes


def test_reload_from_disk_and_predict_folder(tiny_model_dir, image_folder):
    predictor = Predictor.load(tiny_model_dir)
    assert predictor.image_size == 32 and predictor.policy.animal_threshold == 0.5

    preds = predictor.predict_folder(image_folder)
    assert [p.filename for p in preds] == ["a.jpg", "b.png", "broken.jpg", "c_gray.jpg"]
    broken = preds[2]
    assert broken.label == "error" and broken.error
    for p in preds[:2] + preds[3:]:
        assert p.label in LABELS
        assert abs(sum(p.probabilities.values()) - 1) < 1e-4
        assert 0 <= p.confidence <= 1


def test_duplicate_names_keep_order(tiny_model_dir):
    predictor = Predictor.load(tiny_model_dir)
    preds = predictor.predict_sources([("x.jpg", jpeg_bytes()), ("x.jpg", b"bad"), ("x.jpg", jpeg_bytes())])
    assert [p.label == "error" for p in preds] == [False, True, False]


def test_table_matches_brief_format(tiny_model_dir, image_folder):
    table = format_table(Predictor.load(tiny_model_dir).predict_folder(image_folder))
    header = table.splitlines()[0]
    assert header.split() == ["filename", "prediction", "confidence"]
    assert "a.jpg" in table and "%" in table
