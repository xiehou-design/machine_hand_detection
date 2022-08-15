import torch
import numpy as np
import h5py
from EMG.model.conv1D import Model
from torch.utils.data import DataLoader
from EMG.dataset import ImageDataset
from torch.utils.tensorboard import SummaryWriter
import time

now_time = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
train_writer = SummaryWriter(log_dir='log/{}/train'.format(now_time))
test_writer = SummaryWriter(log_dir='log/{}/test'.format(now_time))


def load_image_feature(image_feature_filepath):
    '''读取h5数据文件'''
    file = h5py.File(image_feature_filepath, 'r')
    imageData = file['featureData'][:]
    # 放缩
    imageData = imageData / 10000
    imageLabel = file['featureLabel'][:]
    file.close()
    return imageData, imageLabel


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train(True)
    correct = 0
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.type(torch.float32)
        y = y.type(torch.long)
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()
            model = model.cuda()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss_fn(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /= len(dataloader)
    correct /= size
    return train_loss, correct


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.type(torch.float32)
            y = y.type(torch.long)
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
                model = model.cuda()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct


if __name__ == '__main__':
    imageData, imageLabel = load_image_feature('data/sku_data/image_feature.h5')
    # 随机打乱数据和标签
    N = imageData.shape[0]
    index = np.random.permutation(N)
    data = imageData[index, :]
    label = imageLabel[index]

    # 对数据升维,标签one-hot
    data = np.expand_dims(data, axis=2)
    data = np.expand_dims(data, axis=3)
    # label = convert_to_one_hot(label, 16).T

    # 划分数据集
    N = data.shape[0]
    num_train = round(N * 0.8)
    X_train = data[0:num_train, :, :, :]
    Y_train = label[0:num_train]
    X_test = data[num_train:N, :, :, :]
    Y_test = label[num_train:N]

    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))

    batch_size = 8
    learning_rate = 1e-3
    epochs = 100

    train_dataset = ImageDataset(X_train, Y_train)
    test_dataset = ImageDataset(X_test, Y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = Model(classes=16)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=5e-5)

    best_loss = 100
    for epoch in range(1, epochs + 1):
        train_loss, correct = train(train_dataloader, model, loss_function, optimizer)
        train_writer.add_scalar('loss', train_loss, epoch)
        train_writer.add_scalar('acc', correct, epoch)
        test_loss, correct = test(test_dataloader, model, loss_function)
        test_writer.add_scalar('loss', test_loss, epoch)
        test_writer.add_scalar('acc', correct, epoch)
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), './savemodel/epoch_{}_{}_{}.pth'.format(epoch, correct, test_loss))
        lr_scheduler.step()
    # pass
    train_writer.close()
    test_writer.close()
