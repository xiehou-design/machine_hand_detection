import torch
from PIL import Image
from Hand.model_hand_1.model import HandModel
import os
import numpy as np

if __name__ == '__main__':
    model_path = 'savemodel/epoch_5_1.0_0.9060988028844198.pth'
    model = HandModel(classes=5)
    model.load_state_dict(torch.load(model_path))

    for index, image_name in enumerate(os.listdir('../data/hand1')):
        image_filepath = os.path.join('./sku_data/hand1', image_name)
        image = Image.open(image_filepath).convert('RGB')
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
        label = torch.argmax(pred, dim=-1).cpu().numpy()[0]
        print(label, image_name)
        if index == 10:
            break
