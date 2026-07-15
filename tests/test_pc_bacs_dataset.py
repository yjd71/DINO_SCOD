from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode

from utils.dataloader import SelectionPoolDataset


def _write_rgb(path: Path, color=(255, 0, 0), size=(5, 3)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size, color=color).save(path)


def test_selection_pool_reads_rgb_without_gt(tmp_path):
    image_root = tmp_path / 'TR-CAMO' / 'im'
    _write_rgb(image_root / 'sample.png')

    dataset = SelectionPoolDataset(image_root)
    key, image = dataset[0]

    assert key == 'TR-CAMO/sample'
    assert dataset.sample_keys == ['TR-CAMO/sample']
    assert image.shape == (3, 392, 392)
    assert dataset.transform.transforms[0].interpolation is InterpolationMode.BILINEAR
    assert dataset.transform.transforms[0].antialias is True
    expected = torch.tensor([
        (1.0 - 0.485) / 0.229,
        (0.0 - 0.456) / 0.224,
        (0.0 - 0.406) / 0.225,
    ])
    torch.testing.assert_close(image[:, 0, 0], expected)


def test_selection_pool_distinguishes_same_basename_across_subsets(tmp_path):
    camo_root = tmp_path / 'TR-CAMO' / 'im'
    cod10k_root = tmp_path / 'TR-COD10K' / 'im'
    _write_rgb(camo_root / 'shared.jpg')
    _write_rgb(cod10k_root / 'shared.png')

    dataset = SelectionPoolDataset([cod10k_root, camo_root], image_size=8)

    assert dataset.sample_keys == ['TR-CAMO/shared', 'TR-COD10K/shared']


def test_selection_pool_order_is_stable_by_sample_key(tmp_path):
    camo_root = tmp_path / 'TR-CAMO' / 'im'
    cod10k_root = tmp_path / 'TR-COD10K' / 'im'
    _write_rgb(camo_root / 'zebra.png')
    _write_rgb(camo_root / 'ant.png')
    _write_rgb(cod10k_root / 'middle.png')

    forward = SelectionPoolDataset([camo_root, cod10k_root], image_size=8)
    reverse = SelectionPoolDataset([cod10k_root, camo_root], image_size=8)

    expected = ['TR-CAMO/ant', 'TR-CAMO/zebra', 'TR-COD10K/middle']
    assert forward.sample_keys == expected
    assert reverse.sample_keys == expected
    assert [forward[index][0] for index in range(len(forward))] == expected


def test_selection_pool_read_error_contains_key_and_path(tmp_path):
    image_root = tmp_path / 'TR-CAMO' / 'im'
    broken_path = image_root / 'broken.jpg'
    broken_path.parent.mkdir(parents=True)
    broken_path.write_bytes(b'not an image')
    dataset = SelectionPoolDataset(image_root, image_size=8)

    with pytest.raises(FileNotFoundError) as error:
        dataset[0]

    message = str(error.value)
    assert 'TR-CAMO/broken' in message
    assert str(broken_path) in message
