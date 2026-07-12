import os
import torch
from numbers import Integral
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from utils.data_augmentation import cv_random_flip, randomCrop, randomRotation, randomPeper, colorEnhance
import cv2


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
MASK_EXTENSIONS = ('.png', '.jpg', '.jpeg')


def _as_list(paths):
    if isinstance(paths, (list, tuple)):
        return list(paths)
    return [paths]


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _sample_key(*parts):
    return '/'.join(str(part).strip('/\\') for part in parts if str(part).strip('/\\'))


def _normalize_sample_key(value):
    return _sample_key(str(value).replace('\\', '/'))


def _subset_name_from_image_root(image_root):
    leaf = os.path.basename(os.path.normpath(image_root))
    if leaf.lower() in {'im', 'imgs', 'images'}:
        return os.path.basename(os.path.dirname(os.path.normpath(image_root)))
    return leaf


def _validate_sample_keys(sampled, items, source_path):
    available_keys = {item['key'] for item in items}
    available_stems = {item['stem'] for item in items}
    missing = sorted(key for key in sampled if key not in available_keys and key not in available_stems)
    if missing:
        preview = '\n'.join(missing[:5])
        raise ValueError(f'>>> {len(missing)} sampled entries from {source_path} do not match training images, examples:\n{preview}')


def _load_txt_sample_keys(txt_path, items):
    with open(txt_path, 'r') as f:
        sampled = {_normalize_sample_key(line.strip()) for line in f if line.strip()}
    _validate_sample_keys(sampled, items, txt_path)
    return sampled


def _load_pt_sample_keys(pt_path, items):
    sampled = torch.load(pt_path, map_location='cpu')
    if torch.is_tensor(sampled):
        values = sampled.detach().cpu().flatten().tolist()
    elif isinstance(sampled, (list, tuple, set)):
        values = list(sampled)
    else:
        raise TypeError(f'>>> Unsupported labeled indices format in {pt_path}: {type(sampled).__name__}')

    if not values:
        return set()

    if all(isinstance(value, str) for value in values):
        sampled_keys = {_normalize_sample_key(value) for value in values if str(value).strip()}
        _validate_sample_keys(sampled_keys, items, pt_path)
        return sampled_keys

    if all(isinstance(value, Integral) and not isinstance(value, bool) for value in values):
        sample_count = len(items)
        indices = [int(value) for value in values]
        out_of_range = [index for index in indices if index < 0 or index >= sample_count]
        if out_of_range:
            preview = '\n'.join(str(index) for index in out_of_range[:5])
            raise IndexError(f'>>> {len(out_of_range)} labeled indices from {pt_path} are outside [0, {sample_count - 1}], examples:\n{preview}')
        return {items[index]['key'] for index in indices}

    raise TypeError(f'>>> Unsupported labeled indices values in {pt_path}; expected all strings or all integers.')


def _load_sample_keys(txt_path, indices_pt, items):
    if indices_pt is not None:
        return _load_pt_sample_keys(indices_pt, items)
    if txt_path is not None:
        return _load_txt_sample_keys(txt_path, items)
    return None


def _find_matching_file(root, basename, extensions):
    for ext in extensions:
        path = os.path.join(root, basename + ext)
        if os.path.exists(path):
            return path
    return None


def _collect_images(image_roots):
    images = []
    for image_root in _as_list(image_roots):
        subset = _subset_name_from_image_root(image_root)
        for filename in sorted(os.listdir(image_root)):
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                basename = os.path.splitext(filename)[0]
                images.append({
                    'key': _sample_key(subset, basename),
                    'stem': basename,
                    'image': os.path.join(image_root, filename),
                })
    return sorted(images, key=lambda item: item['key'])


def _collect_labeled_pairs(image_roots, gt_roots):
    image_roots = _as_list(image_roots)
    gt_roots = _as_list(gt_roots)
    assert len(image_roots) == len(gt_roots), '>>> Number of image roots and gt roots do not match.'

    pairs = []
    missing = []
    for image_root, gt_root in zip(image_roots, gt_roots):
        subset = _subset_name_from_image_root(image_root)
        for filename in sorted(os.listdir(image_root)):
            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue
            basename = os.path.splitext(filename)[0]
            gt_path = _find_matching_file(gt_root, basename, MASK_EXTENSIONS)
            if gt_path is None:
                missing.append(os.path.join(gt_root, basename + '.png'))
                continue
            pairs.append({
                'key': _sample_key(subset, basename),
                'stem': basename,
                'image': os.path.join(image_root, filename),
                'gt': gt_path,
            })

    if missing:
        preview = '\n'.join(missing[:5])
        raise FileNotFoundError(f'>>> Missing GT masks for {len(missing)} images, examples:\n{preview}')
    return sorted(pairs, key=lambda pair: pair['key'])


def _collect_masks(mask_roots):
    masks = {}
    stem_paths = {}
    stem_counts = {}
    for mask_root in _as_list(mask_roots):
        for dirpath, _, filenames in os.walk(mask_root):
            rel_dir = os.path.relpath(dirpath, mask_root)
            rel_dir = '' if rel_dir == '.' else rel_dir
            parent = os.path.basename(os.path.normpath(dirpath))
            for filename in filenames:
                if not filename.lower().endswith(MASK_EXTENSIONS):
                    continue
                basename = os.path.splitext(filename)[0]
                path = os.path.join(dirpath, filename)
                if rel_dir:
                    masks.setdefault(_sample_key(rel_dir, basename), path)
                masks.setdefault(_sample_key(parent, basename), path)
                stem_paths.setdefault(basename, path)
                stem_counts[basename] = stem_counts.get(basename, 0) + 1
    for basename, path in stem_paths.items():
        if stem_counts[basename] == 1:
            masks[basename] = path
    return masks


class LabeledTrainDataset(Dataset):
    def __init__(self, l_image_root,  # file path: string, Original RGB Images
                       l_gt_root,  # file path: string, GT Annotations
                       l_txt_root,  # txt file path: string, Sampled Images
                       l_train_size,
                       labeled_indices_pt=None,
                       rVFlip=False,
                       rCrop=False,
                       rRotate=False,
                       colorEnhance=False,
                       rPeper=False,
                       return_sample_key=False):

        self.l_train_size = l_train_size
        self.patch_size = 14

        # data augmentation
        self.rVFlip = rVFlip
        self.rCrop = rCrop
        self.rRotate = rRotate
        self.colorEnhance = colorEnhance
        self.rPeper = rPeper
        self.return_sample_key = bool(return_sample_key)

        # load labeled data
        self.l_images, self.l_gts = [], []
        pairs = _collect_labeled_pairs(l_image_root, l_gt_root)
        sampled = _load_sample_keys(l_txt_root, labeled_indices_pt, pairs)
        if sampled is not None:
            pairs = [pair for pair in pairs if pair['key'] in sampled or pair['stem'] in sampled]

        self.l_images = [pair['image'] for pair in pairs]
        self.l_gts = [pair['gt'] for pair in pairs]
        self.sample_keys = [pair['key'] for pair in pairs]


        assert len(self.l_images) == len(self.l_gts), '>>> Number of labeled images and gts do not match.'

        self.size = len(self.l_images)

        for i in range(self.size):
            assert _stem(self.l_images[i]) == _stem(self.l_gts[i]), '>>> File name mismatch.'

        # transforms
        self.img_transforms = transforms.Compose([
            transforms.Resize((self.l_train_size, self.l_train_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.gt_transforms = transforms.Compose([
            transforms.Resize((self.l_train_size, self.l_train_size)),
            transforms.ToTensor()])

        print(f'>>> Training/Validating with {self.size} samples')

    def __getitem__(self, index):
        l_image = self.rgb_loader(self.l_images[index])
        l_gt = self.binary_loader(self.l_gts[index])

        # data augmentation
        if self.rVFlip:
            l_image, l_gt = cv_random_flip([l_image, l_gt])
        if self.rCrop:
            l_image, l_gt = randomCrop([l_image, l_gt])
        if self.rRotate:
            l_image, l_gt = randomRotation([l_image, l_gt])
        if self.colorEnhance:
            l_image = colorEnhance(l_image)
        if self.rPeper:
            l_gt = randomPeper(l_gt)

        ori_img = self.gt_transforms(l_image)  # used for seg evaluate

        # labeled data processing
        l_image = self.img_transforms(l_image)
        # l_image_fold = self.image_fold(l_image)
        l_gt = self.gt_transforms(l_gt)

        if self.return_sample_key:
            return ori_img, l_image, l_gt, self.sample_keys[index]
        return ori_img, l_image, l_gt

    def __len__(self):
        return self.size

    def rgb_loader(self, path):
        img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f'Failed to read RGB image: {path}')
        img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img, mode='RGB')
        return img

    def binary_loader(self, path):
        img = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f'Failed to read mask: {path}')
        img = Image.fromarray(img, mode='L')
        return img


class PCLabeledTrainDataset(LabeledTrainDataset):
    """Labeled training data with a stable image id appended to each sample.

    The legacy :class:`LabeledTrainDataset` keeps returning exactly
    ``(original_image, normalized_image, gt)`` by default.  PC-HBM training
    opts into this subclass and receives ``(..., sample_key)`` so retrieval
    can exclude a query image from its own labeled memory.
    """

    def __init__(self, *args, **kwargs):
        kwargs["return_sample_key"] = True
        super().__init__(*args, **kwargs)


class LabeledMemoryDataset(LabeledTrainDataset):
    """Deterministic labeled-only dataset used to rebuild PC-HBM memory.

    Samples are ``(sample_key, normalized_image, gt)``.  There is no random
    crop, flip, rotation, colour enhancement, or mask noise, and masks use
    nearest-neighbour resize to preserve their binary semantics.
    """

    def __init__(
        self,
        l_image_root,
        l_gt_root,
        l_txt_root,
        l_train_size,
        labeled_indices_pt=None,
    ):
        super().__init__(
            l_image_root=l_image_root,
            l_gt_root=l_gt_root,
            l_txt_root=l_txt_root,
            l_train_size=l_train_size,
            labeled_indices_pt=labeled_indices_pt,
            rVFlip=False,
            rCrop=False,
            rRotate=False,
            colorEnhance=False,
            rPeper=False,
            return_sample_key=False,
        )
        self.gt_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (self.l_train_size, self.l_train_size),
                    interpolation=InterpolationMode.NEAREST,
                ),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        image = self.rgb_loader(self.l_images[index])
        gt = self.binary_loader(self.l_gts[index])
        return self.sample_keys[index], self.img_transforms(image), self.gt_transforms(gt)


def build_labeled_memory_loader(
    l_image_root,
    l_gt_root,
    l_txt_root,
    l_train_size,
    *,
    labeled_indices_pt=None,
    batch_size=16,
    num_workers=0,
    pin_memory=True,
):
    """Build the full, ordered, non-distributed labeled memory loader."""

    dataset = LabeledMemoryDataset(
        l_image_root=l_image_root,
        l_gt_root=l_gt_root,
        l_txt_root=l_txt_root,
        l_train_size=l_train_size,
        labeled_indices_pt=labeled_indices_pt,
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(num_workers),
    )


class UnlabeledTrainDataset(Dataset):
    def __init__(self, u_image_root,  # file path: string
                       u_gt_root,  # file path: string
                       sampled_txt,
                       u_train_size,
                       labeled_indices_pt=None,
                       rVFlip=True,
                       rCrop=True,
                       rRotate=False,
                       colorEnhance=True,
                       rPeper=False):

        self.u_train_size = u_train_size

        self.patch_size = 14

        # data augmentation
        self.rVFlip = rVFlip
        self.rCrop = rCrop
        self.rRotate = rRotate
        self.colorEnhance = colorEnhance
        self.rPeper = rPeper

        # unlabeled data
        self.u_images, self.u_gts = [], []
        images = _collect_images(u_image_root)
        self.sampled = _load_sample_keys(sampled_txt, labeled_indices_pt, images) or set()
        masks = _collect_masks(u_gt_root)

        missing = []
        for item in images:
            if item['key'] in self.sampled or item['stem'] in self.sampled:
                continue
            gt_path = masks.get(item['key']) or masks.get(item['stem'])
            if gt_path is None:
                missing.append(item['key'])
                continue
            self.u_images.append(item['image'])
            self.u_gts.append(gt_path)

        if missing:
            preview = '\n'.join(missing[:5])
            raise FileNotFoundError(f'>>> Missing SAM masks for {len(missing)} unlabeled images, examples:\n{preview}')
        
        pairs = sorted(zip(self.u_images, self.u_gts), key=lambda pair: _sample_key(_stem(pair[0])))
        self.u_images = [pair[0] for pair in pairs]
        self.u_gts = [pair[1] for pair in pairs]

        assert len(self.u_images) == len(self.u_gts), '>>> Number of unlabeled images and gts do not match.'

        self.size = len(self.u_images)

        for i in range(self.size):
            assert _stem(self.u_images[i]) == _stem(self.u_gts[i]), '>>> Unlabeled image and gt do not match.'

        # transforms
        self.img_transforms = transforms.Compose([
            transforms.Resize((self.u_train_size, self.u_train_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.gt_transforms = transforms.Compose([
            transforms.Resize((self.u_train_size, self.u_train_size)),
            transforms.ToTensor()])

        print(f'>>> Training/Validating with {self.size} samples')

    def __getitem__(self, index):
        u_image = self.rgb_loader(self.u_images[index])
        u_gt = self.binary_loader(self.u_gts[index])

        # data augmentation
        if self.rVFlip:
            u_image, u_gt = cv_random_flip([u_image, u_gt])
        if self.rCrop:
            u_image, u_gt = randomCrop([u_image, u_gt])
        if self.rRotate:
            u_image, u_gt = randomRotation([u_image, u_gt])
        if self.colorEnhance:
            u_image = colorEnhance(u_image)
        if self.rPeper:
            u_gt = randomPeper(u_gt)

        # unlabeled data processing
        u_image = self.img_transforms(u_image)
        u_gt = self.gt_transforms(u_gt)

        return u_image, u_gt

    def __len__(self):
        return self.size

    def rgb_loader(self, path):
        img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f'Failed to read RGB image: {path}')
        img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img, mode='RGB')
        return img

    def binary_loader(self, path):
        img = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f'Failed to read mask: {path}')
        img = Image.fromarray(img, mode='L')
        return img


class UnlabeledPseudoTrainDataset(Dataset):
    """Unlabeled images for online teacher-generated pseudo labels.

    This dataset deliberately has no mask-root argument: pseudo targets are
    generated by the teacher in the trainer after image augmentation.
    """

    def __init__(self, u_image_root,
                 sampled_txt,
                 u_train_size,
                 labeled_indices_pt=None,
                 rVFlip=True,
                 rCrop=True,
                 rRotate=False,
                 colorEnhance=True):
        self.u_train_size = u_train_size
        self.rVFlip = rVFlip
        self.rCrop = rCrop
        self.rRotate = rRotate
        self.colorEnhance = colorEnhance

        images = _collect_images(u_image_root)
        sampled = _load_sample_keys(sampled_txt, labeled_indices_pt, images) or set()
        self.u_images = [
            item['image']
            for item in images
            if item['key'] not in sampled and item['stem'] not in sampled
        ]
        self.size = len(self.u_images)

        self.img_transforms = transforms.Compose([
            transforms.Resize((self.u_train_size, self.u_train_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print(f'>>> Teacher-pseudo training with {self.size} unlabeled samples')

    def __getitem__(self, index):
        u_image = self.rgb_loader(self.u_images[index])

        if self.rVFlip:
            u_image = cv_random_flip([u_image])[0]
        if self.rCrop:
            u_image = randomCrop([u_image])[0]
        if self.rRotate:
            u_image = randomRotation([u_image])[0]
        if self.colorEnhance:
            u_image = colorEnhance(u_image)

        return self.img_transforms(u_image)

    def __len__(self):
        return self.size

    def rgb_loader(self, path):
        img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f'Failed to read RGB image: {path}')
        img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
        return Image.fromarray(img, mode='RGB')


class TestDataset(Dataset):
    def __init__(self, image_root,  # file path: string
                       gt_root,  # file path: string
                       test_size):
        
        self.test_size = test_size

        pairs = _collect_labeled_pairs(image_root, gt_root)
        self.images = [pair['image'] for pair in pairs]
        self.gts = [pair['gt'] for pair in pairs]

        assert len(self.images) == len(self.gts), '>>> Number of labeled images and gts do not match.'

        self.size = len(self.images)

        for i in range(self.size):
            assert _stem(self.images[i]) == _stem(self.gts[i]), '>>> File name mismatch.'


        # transforms
        self.img_transforms = transforms.Compose([
            transforms.Resize((self.test_size, self.test_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.gt_transforms = transforms.Compose([
            transforms.Resize((self.test_size, self.test_size)),
            transforms.ToTensor()])

        print(f'>>> Testing/Validating with {self.size} samples')
        
    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        ori_img = self.gt_transforms(image)
        image = self.img_transforms(image)
        ori_gt = transforms.PILToTensor()(gt)
        gt = self.gt_transforms(gt)

        name = os.path.basename(self.images[index])
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        return ori_img, ori_gt, name, image, gt

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f'Failed to read RGB image: {path}')
        img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img, mode='RGB')
        return img

    def binary_loader(self, path):
        img = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f'Failed to read mask: {path}')
        img = Image.fromarray(img, mode='L')
        return img
    
