import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx, 1])
        image = Image.open(img_name).convert('RGB')
        description = self.dataframe.iloc[idx, 0]
        encoded_text = self.tokenizer(description, return_tensors="pt", padding='max_length', truncation=True, max_length=64)
        sample = {'image': image, 'encoded_text': encoded_text['input_ids'].squeeze()}
        
        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample
