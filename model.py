import torch
import torch.nn as nn
import predict_LSTM


class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class Classifier(nn.Module):
    def __init__(self, input):
        super(Classifier, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock1D(64)
        self.block3 = ResidualBlock1D(64)
        self.block4 = ResidualBlock1D(64)
        self.block5 = nn.Sequential(
            nn.Conv1d(64, 1, kernel_size=3, padding=1),
            nn.PReLU()
        )
        self.block6 = nn.Sequential(
            nn.Linear(input, 256), nn.PReLU(), nn.BatchNorm1d(1),
            nn.Linear(256, 512), nn.PReLU(), nn.BatchNorm1d(1),
            nn.Linear(512, 1024), nn.PReLU(), nn.BatchNorm1d(1),
            nn.Linear(1024, 512), nn.PReLU(), nn.BatchNorm1d(1),
            nn.Linear(512, 128), nn.PReLU(), nn.BatchNorm1d(1),
            nn.Linear(128, 6), nn.Sigmoid()
        )

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block1 + block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        out = torch.squeeze(block6)
        return out


# netC = Classifier(100)
# trainx, trainy, testx, tesety = predict_LSTM.data_tensor()
# trainx = torch.unsqueeze(trainx,1)
# print(trainx.shape)
# out = netC(trainx)
# print(out.shape)
