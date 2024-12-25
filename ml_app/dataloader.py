import os
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, data_path, raw_path, top_path, bottom_path):
        self.data_path = data_path
        self.raw_path = raw_path
        self.top_path = top_path
        self.bottom_path = bottom_path
        self.transform = transforms.Compose([  transforms.Resize((576, 576)), 
                                                transforms.ToTensor(), 
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])

    def __len__(self):
        return len(os.listdir(self.top_path))

    def __getitem__(self, idx):
        if random.randint(0,1) == 0:
            top_path = os.path.join(self.top_path, os.listdir(self.top_path)[idx])
            bottom_path = os.path.join(self.bottom_path, os.listdir(self.bottom_path)[idx])
            top_image = self.transform(Image.open(top_path))
            bottom_image = self.transform(Image.open(bottom_path))
            targets = 1
        else:
            top_path = os.path.join(self.top_path, os.listdir(self.top_path)[idx])            
            idx_rand = random.choice(list(range(82)))
            while idx_rand == idx:
                idx_rand = random.choice(list(range(82)))
            bottom_path = os.path.join(self.bottom_path, os.listdir(self.bottom_path)[idx_rand])
            top_image = self.transform(Image.open(top_path))
            bottom_image = self.transform(Image.open(bottom_path))
            targets = 0
        return top_image, bottom_image, targets