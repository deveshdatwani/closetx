import os
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self):
        self.top_path = "/home/deveshdatwani/closetx/ml_app/models/dataset/positive/top"
        self.transform = transforms.Compose([   
                                            transforms.Resize((276, 276)), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])
        self.counter = 1

    def __len__(self):
        return len(os.listdir(self.top_path))

    def __getitem__(self, idx):
        top_base_url = "/home/deveshdatwani/closetx/ml_app/models/dataset/positive/top"
        bottom_base_url = "/home/deveshdatwani/closetx/ml_app/models/dataset/positive/bottom"
        top_path = os.path.join(bottom_base_url, os.listdir(top_base_url)[idx])
        bottom_path = os.path.join(top_base_url, os.listdir(bottom_base_url)[idx])
        top_image = self.transform(Image.open(top_path))
        bottom_image = self.transform(Image.open(bottom_path))
        idx_rand = random.randint(0, len(os.listdir(bottom_base_url)))
        while idx_rand == idx:
            idx_rand = random.randint(0, len(os.listdir(bottom_base_url)))
        wrong_bottom_path = os.path.join(bottom_base_url, os.listdir(bottom_base_url)[idx_rand])
        wrong_bottom_image = self.transform(Image.open(wrong_bottom_path))
        return top_image, bottom_image, wrong_bottom_image