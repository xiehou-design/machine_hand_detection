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

        self.label2id = {
            'ok': 0,
            'yes': 1,
            'eight': 2,
            'five': 3,
            'four': 4,
            'three': 5
        }
        self.image_names = []
        with open(txt_filepath, 'r', encoding='utf-8') as file:
            for line in file:
                self.image_names.append(line.strip())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        image_name = self.image_names[item]
        label = None
        for name in self.label2id.keys():
            if name in image_name:
                label = self.label2id[name]
                break

        image_filepath = os.path.join(self.image_dir, image_name)
        image = Image.open(image_filepath).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == '__main__':
    train_transform = transforms.Compose([
        transforms.RandomRotation((-30, 30)),
        transforms.Resize((400, 400)),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop((400, 400)),
        transforms.ToTensor()
    ])
    import matplotlib.pyplot as plt
    import numpy as np

    handDataset = HandDataset('../data/hand2', '../data/test.txt', train_transform)
    for image, label in handDataset:
        print(image.shape)
        image = np.transpose(image, (1, 2, 0))
        plt.imshow(image)
        plt.show()
        print(image)
        print(label)
        break
