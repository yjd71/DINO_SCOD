import sys
import threading

import cv2
import numpy as np
import pytest

import evaluate as evaluate_module


class _RecordingMetric:
    instances = []

    def __init__(self):
        self.values = []
        self.step_threads = []
        type(self).instances.append(self)

    def step(self, pred, gt):
        self.values.append(int(pred[0, 0]))
        self.step_threads.append(threading.get_ident())

    def get_results(self):
        return {'order': tuple(self.values), 'total': sum(self.values)}


@pytest.fixture
def image_folders(tmp_path):
    pred_root = tmp_path / 'predictions'
    mask_root = tmp_path / 'masks'
    pred_root.mkdir()
    mask_root.mkdir()
    for name, value in [('b.png', 20), ('a.png', 10), ('c.png', 30)]:
        pred = np.full((5, 7), value, dtype=np.uint8)
        mask = np.full((5, 7), 255 - value, dtype=np.uint8)
        assert cv2.imwrite(str(pred_root / name), pred)
        assert cv2.imwrite(str(mask_root / name), mask)
    return pred_root, mask_root


def test_parallel_reading_preserves_metric_results_and_sorted_step_order(
    image_folders, monkeypatch
):
    pred_root, mask_root = image_folders
    _RecordingMetric.instances.clear()
    monkeypatch.setattr(evaluate_module, 'EvaluationMetricsV2', _RecordingMetric)

    sequential = evaluate_module.evaluate(str(pred_root), str(mask_root), workers=0)
    parallel = evaluate_module.evaluate(
        str(pred_root), str(mask_root), workers=2, prefetch=2
    )

    assert sequential == parallel == {'order': (10, 20, 30), 'total': 60}
    main_thread = threading.get_ident()
    assert _RecordingMetric.instances[0].step_threads == [main_thread] * 3
    assert _RecordingMetric.instances[1].step_threads == [main_thread] * 3


@pytest.mark.parametrize('workers', [0, 2])
def test_read_failure_reports_the_missing_gt_path(image_folders, monkeypatch, workers):
    pred_root, mask_root = image_folders
    missing_path = mask_root / 'b.png'
    missing_path.unlink()
    monkeypatch.setattr(evaluate_module, 'EvaluationMetricsV2', _RecordingMetric)

    with pytest.raises(FileNotFoundError, match=r'Failed to read GT mask:.*b\.png'):
        evaluate_module.evaluate(
            str(pred_root), str(mask_root), workers=workers, prefetch=2
        )


def test_shape_mismatch_reports_both_paths(image_folders, monkeypatch):
    pred_root, mask_root = image_folders
    assert cv2.imwrite(
        str(mask_root / 'a.png'), np.zeros((3, 4), dtype=np.uint8)
    )
    monkeypatch.setattr(evaluate_module, 'EvaluationMetricsV2', _RecordingMetric)

    with pytest.raises(ValueError, match=r'Prediction and GT shapes do not match:.*') as exc:
        evaluate_module.evaluate(str(pred_root), str(mask_root), workers=2, prefetch=2)

    message = str(exc.value)
    assert str(pred_root / 'a.png') in message
    assert str(mask_root / 'a.png') in message


def test_evaluate_loader_options_are_validated(image_folders):
    pred_root, mask_root = image_folders

    with pytest.raises(ValueError, match='workers must be a non-negative integer'):
        evaluate_module.evaluate(str(pred_root), str(mask_root), workers=-1)
    with pytest.raises(ValueError, match='prefetch must be a positive integer'):
        evaluate_module.evaluate(
            str(pred_root), str(mask_root), workers=1, prefetch=0
        )


def test_parse_args_exposes_parallel_reading_controls(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['evaluate.py'])
    defaults = evaluate_module.parse_args()
    assert defaults.workers == 4
    assert defaults.prefetch == 8

    monkeypatch.setattr(
        sys,
        'argv',
        ['evaluate.py', '--workers', '0', '--prefetch', '3'],
    )
    explicit = evaluate_module.parse_args()
    assert explicit.workers == 0
    assert explicit.prefetch == 3

    monkeypatch.setattr(sys, 'argv', ['evaluate.py', '--workers', '-1'])
    with pytest.raises(SystemExit):
        evaluate_module.parse_args()
