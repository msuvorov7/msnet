from typing import Any

import torch
import torch.nn as nn


class SigmoidFunction(torch.autograd.Function):
    """
    Функция для реализации прямого и обратного прохода
    функции активации Sigmoid слоя нейронной сети
    """
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Реализация прямого прохода Sigmoid(X) = 1 + (1 - exp(-X), где операция exp
        применяется поэлементно
        :param ctx: контекст для сохранения производных при прямом обходе
        :param args: аргументы в порядке [X,]
        :param kwargs:
        :return: значения сигмоиды от входа
        """
        inputs = args[0]
        sigmoid = 1 / (1 + torch.exp(-inputs))
        ctx.save_for_backward(sigmoid)
        return sigmoid

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """
        Обратное распростанение градиента для Sigmoid(X) = 1 + (1 - exp(-X)

        dL/dX = dL/dz * dz/dX = dL/dz * sigmoid * (1 - sigmoid), где sigmoid - функция
        сигмоиды от X

        :param ctx: контекст с сохраненными значениями при прямом проходе
        :param grad_outputs: приходящий градиент
        :return: градиенты по dX
        """
        sigmoid = ctx.saved_tensors[0]
        return grad_outputs[0] * sigmoid * (1 - sigmoid)


class Sigmoid(nn.Module):
    """
    Реализация функции активации Sigmoid слоя нейронной сети.
    Для переопределения метода backward требуется создать свою функцию
    и вызывать ее методы.
    """
    def __init__(self):
        super().__init__()
        self.sigmoid = SigmoidFunction.apply

    def forward(self, x):
        return self.sigmoid(x)
