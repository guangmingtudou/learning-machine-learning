import torch
import MyTool

true_w = torch.tensor([3.4, 4])
true_b = 5

def synthetic_data(W, B, data_number):
    X = torch.normal(0, 1, size=(data_number, 2))
    y = torch.matmul(X, W) + B
    y += torch.normal(0, 1, size=y.shape)
    return X, y

features, labels = synthetic_data(true_w, true_b, 1000)
#MyTool.plot_points_3d(features[:, 0].tolist(), features[:, 1].tolist(), labels.tolist())

def data_iter(features, labels, patch_size):
    l = features.size(axis=0)
    present_index=0
    while present_index < l:
        yield features[present_index:min(present_index+patch_size, l)], labels[present_index:min(present_index+patch_size, l)]
        present_index += patch_size

def train_epoch(features, labels, w, b, lr):
    batch_size = 10
    for X, y in data_iter(features, labels, batch_size):
        y_hat = torch.matmul(X, w) + b
        l = (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
        l.sum().backward()
        with torch.no_grad():
            w -=lr*w.grad/batch_size
            b -=lr*b.grad/batch_size
            w.grad.zero_()
            b.grad.zero_()

def train(features, labels, w, b, lr, epoch_number, batch_size):
    for epoch in range(epoch_number):
        train_epoch(features, labels, w, b, lr)
        print(f'第{epoch+1}轮训练,w正确率:{1-(abs(w-true_w)/true_w)},b正确率:{1-(abs(b-true_b)/true_b)}')

w = torch.normal(0, 1, size=true_w.shape, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

train(features, labels, w, b, lr=0.01, epoch_number=3, batch_size = 10)