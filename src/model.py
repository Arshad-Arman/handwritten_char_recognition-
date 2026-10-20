"""Neural network architectures: CNN (single character) and CRNN (word / sequence)."""
import torch.nn as nn


class CNN(nn.Module):
    """VGG-style CNN for 28x28 grayscale characters.

    3 blocks of [Conv-BN-ReLU] x2 + MaxPool  ->  28 -> 14 -> 7 -> 3,
    then a 2-layer classifier with dropout.
    """

    def __init__(self, num_classes: int, dropout: float = 0.4):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(128 * 3 * 3, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _conv_bn(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network trained with CTC loss.

    Input : (B, 1, 32, W)  grayscale word image (height fixed to 32)
    Output: (T, B, num_classes + 1) logits with T = W // 4 and class 0 = CTC blank.

    CNN  : extracts a feature column for every 4 horizontal pixels (height collapsed to 1)
    BiLSTM: models left/right context between the columns
    Linear: per-time-step character scores
    """

    def __init__(self, num_classes: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            _conv_bn(1, 32), nn.MaxPool2d(2, 2),                       # 32xW   -> 16xW/2
            _conv_bn(32, 64), nn.MaxPool2d(2, 2),                      #        -> 8xW/4
            _conv_bn(64, 128), _conv_bn(128, 128), nn.MaxPool2d((2, 1), (2, 1)),   # -> 4xW/4
            _conv_bn(128, 256), nn.MaxPool2d((2, 1), (2, 1)),          #        -> 2xW/4
            nn.Conv2d(256, 256, (2, 1)), nn.ReLU(inplace=True),        #        -> 1xW/4
        )
        self.rnn = nn.LSTM(256, hidden, num_layers=2, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden * 2, num_classes + 1)

    def forward(self, x):
        f = self.cnn(x).squeeze(2)          # B, C, T
        f = f.permute(2, 0, 1)              # T, B, C
        out, _ = self.rnn(f)
        return self.fc(out)                 # T, B, K
