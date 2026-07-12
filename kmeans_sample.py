import cv2
from PIL import Image
import os
import torch
import numpy as np
from sklearn.cluster import KMeans
from torchvision import transforms
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Select representative RSBL training samples with DINOv2 features.')
    parser.add_argument('--data-root', default='./Dataset/COD')
    parser.add_argument('--train-sets', nargs='+', default=['TR-CAMO', 'TR-COD10K'])
    parser.add_argument('--features-path', default='./dino_b_features.npy')
    parser.add_argument('--sampled-txt', default=None)
    parser.add_argument('--n-clusters', type=int, default=40)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=2025)
    return parser.parse_args()

def rgb_loader(path):
    img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Failed to read RGB image: {path}')
    img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img, mode='RGB')
    return img


if __name__ == '__main__':
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError('--batch-size must be positive.')
    data_root = args.data_root
    train_sets = args.train_sets
    img_items = []
    for subset in train_sets:
        data_path = os.path.join(data_root, subset, 'im')
        img_items.extend([
            {
                'path': os.path.join(data_path, img_name),
                'key': f'{subset}/{os.path.splitext(img_name)[0]}',
            }
            for img_name in sorted(os.listdir(data_path))
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
    if not img_items:
        raise ValueError(f'No training images found under: {data_root}')
    if args.n_clusters <= 0 or args.n_clusters > len(img_items):
        raise ValueError(f'--n-clusters must be in [1, {len(img_items)}], got {args.n_clusters}.')
    if args.top_k <= 0:
        raise ValueError('--top-k must be positive.')
    features_save_path = args.features_path
    sampled_txt_path = args.sampled_txt or os.path.join(data_root, 'sampled_images.txt')

    img_transforms = transforms.Compose([
        transforms.Resize((392, 392)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    dino = torch.hub.load('./dinov2', 'dinov2_vitb14', source='local', pretrained=False)
    dino.load_state_dict(torch.load('./dinov2_vitb14_pretrain.pth', map_location='cpu'))
    dino = dino.to('cuda')
    dino.eval()

    features = []
    batch = []

    for item in tqdm(img_items):
        img = rgb_loader(item['path'])
        img = img_transforms(img)
        batch.append(img)
        if len(batch) == args.batch_size:
            imgs = torch.stack(batch, dim=0).to('cuda')
            with torch.inference_mode():
                feat = dino(imgs)
            features.append(feat.cpu().numpy())
            batch = []

    if batch:
        imgs = torch.stack(batch, dim=0).to('cuda')
        with torch.inference_mode():
            feat = dino(imgs)
        features.append(feat.cpu().numpy())

    features = np.concatenate(features, axis=0)
    np.save(features_save_path, features)
    print(f'Features saved to: {features_save_path}')

    features = np.load(features_save_path)
    print(f'Features loaded from: {features_save_path}')

    random_seed = args.seed
    print(f"Random seed: {random_seed}")
    n_clusters = args.n_clusters
    top_k = args.top_k  # for each cluster, select top k samples closest to the cluster center
    print("Running KMeans clustering...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed).fit(features)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    selected_indices = set()

    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        cluster_features = features[cluster_indices]

        # calculate distances to cluster center
        distances = np.linalg.norm(cluster_features - centers[i], axis=1)
        sorted_cluster_indices = cluster_indices[np.argsort(distances)]

        # select top k samples closest to the cluster center
        topk_indices = sorted_cluster_indices[:top_k]
        selected_indices.update(topk_indices)

    final_indices = [idx for idx in range(len(features)) if idx in selected_indices]

    # Save the selected image names to a text file
    with open(sampled_txt_path, 'w') as f:
        for idx in final_indices:
            f.write(f"{img_items[idx]['key']}\n")
    print(f'Sampled image list saved to: {sampled_txt_path}')
