from Hand.model_hand_2.dataset import HandDataset
from torch.utils.data import DataLoader
from Hand.model_hand_2.model import HandModel
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import time

now_time = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

train_writer = SummaryWriter(log_dir='log/{}/train'.format(now_time))
test_writer = SummaryWriter(log_dir='log/{}/test'.format(now_time))


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
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
    train_transform = transforms.Compose([
        transforms.RandomRotation((-30, 30)),
        transforms.Resize((100, 100)),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop((100, 100)),
        transforms.ToTensor()
    ])
    test_transformer = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.CenterCrop((100, 100)),
        transforms.ToTensor()
    ])

    batch_size = 8
    learning_rate = 1e-3
    epochs = 100

    train_dataset = HandDataset(image_dir='../data/hand2', txt_filepath='../data/train.txt', transform=train_transform)
    test_dataset = HandDataset(image_dir='../data/hand2', txt_filepath='../data/test.txt', transform=test_transformer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = HandModel(classes=6)
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
