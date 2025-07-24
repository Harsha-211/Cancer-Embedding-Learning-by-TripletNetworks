import torch
import random
import os
from PIL import Image
from torch.utils.data import Dataset

class TripletFlatDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_images = {}
        self.classes = []
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            images = [
                os.path.join(class_path, f)
                for f in os.listdir(class_path)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ]
            if len(images) > 1:
                self.class_to_images[class_name] = images
                self.classes.append(class_name)

        self.samples = []
        for class_name, image_paths in self.class_to_images.items():
            for img_path in image_paths:
                self.samples.append((img_path, class_name))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        anchor_path, anchor_class = self.samples[idx]

        positive_candidates = [
            img for img in self.class_to_images[anchor_class]
            if img != anchor_path
        ]
        if not positive_candidates:
            raise ValueError(f"No positive samples available for class '{anchor_class}'")
        positive_path = random.choice(positive_candidates)

        negative_classes = [cls for cls in self.classes if cls != anchor_class]
        if not negative_classes:
            raise ValueError(f"No negative class available for '{anchor_class}'")
        negative_class = random.choice(negative_classes)
        negative_path = random.choice(self.class_to_images[negative_class])

        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(positive_path).convert("RGB")
        negative = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative, anchor_class