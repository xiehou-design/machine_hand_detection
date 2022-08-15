from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms


class HandDataset(Dataset):
    def __init__(self, image_dir, txt_filepath, transform=None):
        super(HandDataset, self).__init__()
        self.txt_filepath = txt_filepath
        self.transform = transform
        self.image_dir = image_dir

        self.image_names = []
        with open(txt_filepath, 'r', encoding='utf-8') as file:
            for line in file:
                self.image_names.append(line.strip())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        image_name = self.image_names[item]
        label = image_name.split('_')[-1].split('.')[0]
        label = int(label)
        image_filepath = os.path.join(self.image_dir, image_name)
        image = Image.open(image_filepath).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == '__main__':
    train_transform = transforms.Compose([
        transforms.RandomRotation((-30, 30)),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop((100, 100)),
        transforms.ToTensor()
    ])
    handDataset = HandDataset('../data/hand1', './sku_data/train.txt', train_transform)
    for image, label in handDataset:
        print(image)
        print(label)
        break
