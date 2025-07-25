from torch.utils.data import Dataset, DataLoader , random_split
from PIL import Image
from torchvision import transforms 
import os 
import numpy as np 
import torch
from configs.config import args
import albumentations as A 
import cv2 as cv
import matplotlib.pyplot as plt 


class FireVideoDataset(Dataset):
    def __init__(self, root_dir , chunk_size=30):
        self.chunk_size = chunk_size
        self.samples = []
        self.transform = A.Compose([
            # A.SmallestMaxSize(max_size=256),  # Equivalent to ShortSideScale(256)
            # A.CenterCrop(height=224, width=224),
            A.Normalize(mean=[0.45]*3, std=[0.225]*3),
        ])
        for class_name, label in [('fire', 1), ('no_fire', 0)]:
            class_dir = os.path.join(root_dir, class_name)
            for chunk_folder in os.listdir(class_dir):
                chunk_path = os.path.join(class_dir, chunk_folder)
                
                if os.path.isdir(chunk_path) and len(os.listdir(chunk_path)) >= chunk_size:
                    self.samples.append((chunk_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk_path, label = self.samples[idx]
        
        image_files = os.listdir(chunk_path)[:self.chunk_size]
         
        frames = []

        for fname in image_files:
            img_path = os.path.join(chunk_path, fname)
            img = Image.open(img_path)
            img_np = np.array(img)

            if img is None:
                continue  
            augmented_img = self.transform(images = [img_np])['images'][0]
            frames.append(augmented_img) 
        np_frame = np.array(frames)
         #    its [B , H , W , C]
        np_frame = np.transpose(np_frame, (3 , 0, 1, 2))  # becomes [C , B , H , W]

        return torch.tensor(np_frame, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
    



def data_loader() : 
    dataset = FireVideoDataset(root_dir = args.rootlpt)
    train_size = int(0.7 *len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


    train_data_loader = DataLoader(train_dataset, batch_size=2,shuffle=True , num_workers=0 , drop_last=True)
    test_data_loader = DataLoader(test_dataset, batch_size=2, shuffle=True , num_workers=0, drop_last=True)
    # subset_size = int(len(dataset) * 0.1)

    # subset_data, _ = random_split(dataset, [subset_size, len(dataset) - subset_size])

    # train_subset_size = int(subset_size * 0.8)
    # test_subset_size = subset_size - train_subset_size

    # sample_train_dataset, sample_test_dataset = random_split(subset_data, [train_subset_size, test_subset_size])

    # # the sample if for test purposes , less sample and epoch see fast result 

    # sample_train_data_loader = DataLoader(sample_train_dataset, batch_size=2, shuffle=True, num_workers=8, drop_last=True)
    # sample_test_data_loader = DataLoader(sample_test_dataset, batch_size=2, shuffle=True, num_workers=8, drop_last=True)
    loader_object = {
        'train_loader': train_data_loader , 
        'test_loader' : test_data_loader, 
        # 'sample_train_loader' : sample_train_data_loader, 
        # 'sample_test_loader' : sample_test_data_loader
    }
    return loader_object
