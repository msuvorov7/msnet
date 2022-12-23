import torch.nn as nn

from src.layers.log_softmax import LogSoftmax
from src.criterions.neg_log_likelihood_loss import NLLLoss


class CrossEntropyLoss(nn.Module):
    """
    Реализация функции потерь CrossEntropyLoss для нейронной сети.
    Для переопределения метода backward требуется создать свою функцию
    и вызывать ее методы.

    В PyTorch реализация в виде: CrossEntropyLoss() = LogSoftmax() + NLLLoss(),
    поэтому критерий принимает выходы от линейного слоя и внутри сам применяет
    к ним LogSoftmax()
    """
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()
        self.nllloss = NLLLoss()

    def forward(self, activation, target):
        return self.nllloss(self.log_softmax(activation), target)
