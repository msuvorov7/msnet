from typing import Any

import torch
import torch.nn as nn


class ReLUFunction(torch.autograd.Function):
    """
    Функция для реализации прямого и обратного прохода
    функции активации ReLU слоя нейронной сети
    """
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Реализация прямого прохода ReLU(X) = max(X, 0), где операция max
        применяется поэлементно
        :param ctx: контекст для сохранения производных при прямом обходе
        :param args: аргументы в порядке [X,]
        :param kwargs:
        :return: значения ReLU от входа
        """
        inputs = args[0]
        ctx.save_for_backward(inputs)
        return torch.max(inputs, torch.zeros_like(inputs))

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """
        Обратное распростанение градиента для ReLU(X) = max(X, 0)

        dL/dX = dL/dz * dz/dX = dL/dz * I[x > 0], где I - функция индикатора

        :param ctx: контекст с сохраненными значениями при прямом проходе
        :param grad_outputs: приходящий градиент
        :return: градиенты по dX
        """
        inputs = ctx.saved_tensors[0]
        mask = torch.where(inputs > 0, torch.ones_like(inputs), torch.zeros_like(inputs))
        return grad_outputs[0] * mask


class ReLU(nn.Module):
    """
    Реализация функции активации ReLU слоя нейронной сети.
    Для переопределения метода backward требуется создать свою функцию
    и вызывать ее методы.
    """
    def __init__(self):
        super().__init__()
        self.relu = ReLUFunction.apply

    def forward(self, x):
        return self.relu(x)
