import torch
from PIL import Image
from Hand.model_hand_1.model import HandModel
import os
import numpy as np
import cv2

if __name__ == '__main__':
    model_path = './savemodel/epoch_69_0.9333333333333333_1.15582674741745.pth'
    model = HandModel(classes=6)
    model.load_state_dict(torch.load(model_path))

    label2id = {
        'ok': 0,
        'yes': 1,
        'eight': 2,
        'five': 3,
        'four': 4,
        'three': 5
    }
    id2label = {}
    for label, id in label2id.items():
        id2label[id] = label

    image_names = []
    with open('../data/test.txt', 'r', encoding='utf-8') as file:
        for line in file:
            image_names.append(line.strip())

    for index, image_name in enumerate(image_names):
        image_filepath = os.path.join('../sku_data/hand2', image_name)
        image = cv2.imread(image_filepath)
        image = cv2.resize(image, (100, 100))
        temp_image = cv2.resize(image, (400, 400))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image, dtype=np.float32)
        image = np.transpose(image, [-1, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image / 255
        image = torch.from_numpy(image)
        if torch.cuda.is_available():
            image = image.cuda()
            model = model.cuda()
        model.eval()
        pred = model(image)
        id = torch.argmax(pred, dim=-1).cpu().numpy()[0]
        print(id, image_name)
        cv2.imshow(id2label[id], temp_image)
        cv2.waitKey(0)
        # if index == 10:
        #     break
