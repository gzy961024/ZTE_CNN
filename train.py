import torch
import torch.nn as nn
import data_utils
import predict_LSTM
import model
from sklearn.metrics import accuracy_score, recall_score

net = nn.Sequential(
    nn.Linear(100, 256), nn.ReLU(), nn.BatchNorm1d(256),
    nn.Linear(256, 512), nn.ReLU(), nn.BatchNorm1d(512),
    nn.Linear(512, 1024), nn.ReLU(), nn.BatchNorm1d(1024),
    nn.Linear(1024, 1024), nn.ReLU(), nn.BatchNorm1d(1024),
    nn.Linear(1024, 1024), nn.ReLU(), nn.BatchNorm1d(1024),
    nn.Linear(1024, 1024), nn.ReLU(), nn.BatchNorm1d(1024),
    nn.Linear(1024, 1024), nn.ReLU(), nn.BatchNorm1d(1024),
    nn.Linear(1024, 256), nn.ReLU(), nn.BatchNorm1d(256),
    nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128),
    nn.Linear(128, 6), nn.Sigmoid()
)


def accuracy(y_pred, y_true):
    y_pred = torch.argmax(y_pred, dim=1)
    acc = (y_pred == y_true).float().mean()
    return acc


def train(net, x, y, testx, testy, num_epoch=1000, device='cuda:0'):
    def init_weight(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weight)
    print('training on ', device)
    net.to(device)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    optimizer = torch.optim.Adam(net.parameters())
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epoch):
        print('start epoch ', epoch)
        net.train()
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        y_hat = net(x)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
        print('epoch %d loss ' % (epoch), l)
        with torch.no_grad():
            testx = testx.to(device)
            testy = testy
            predy = net(testx).cpu()
            predy = torch.argmax(predy, dim=1)
            # epoch_acc = accuracy(predy, testy)
            acc = accuracy_score(testy, predy)
            recall = recall_score(testy, predy, average='macro')

            print('epoch %d accuracy is ' % (epoch), acc.item())
            print('epoch %d recall rate is ' % (epoch), recall.item())
            print('epoch %d Final score is ' % (epoch), 200 * acc * recall / (recall + acc))


# datax, datay, testx, testy = predict_LSTM.data_tensor()
# train(net, datax, datay, testx, testy)
trainx, trainy, testx, testy = predict_LSTM.data_tensor()
netC = model.Classifier(100)
trainx = torch.unsqueeze(trainx, 1)
testx = torch.unsqueeze(testx, 1)
train(netC, trainx, trainy, testx, testy)
