from typing import Any

import torch
import torch.nn as nn


class DropoutFunction(torch.autograd.Function):
    """
    Функция для реализации прямого и обратного прохода
    Dropout слоя нейронной сети
    """
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Реализация прямого прохода Dropout с вероятностью зануления p
        :param ctx: контекст для сохранения производных при прямом обходе
        :param args: аргументы в порядке [X, p]
        :param kwargs:
        :return:
        """
        inputs = args[0]
        probability = args[1]
        dropout_mask = torch.where(torch.rand(inputs.shape) < probability,
                                   torch.zeros_like(inputs),
                                   torch.ones_like(inputs)
                                   )
        ctx.save_for_backward(dropout_mask)
        return inputs * dropout_mask

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """
        Обратное распростанение градиента для Dropout

        dL/dX = dL/dz * dz/dX = dL/dz * dropout_mask

        :param ctx: контекст с сохраненными значениями при прямом проходе
        :param grad_outputs: приходящий градиент
        :return: градиенты по dX
        """
        dropout_mask = ctx.saved_tensors[0]
        return grad_outputs[0] * dropout_mask, None


class Dropout(nn.Module):
    def __init__(self, probability: float = 0.5):
        """
        Реализация Dropout слоя нейронной сети.
        Для переопределения метода backward требуется создать свою функцию
        и вызывать ее методы.
        Зануление нейрона с вероятностью probability
        """
        super().__init__()
        assert (probability >= 0) and (probability < 1)
        self.probability = probability
        self.dropout = DropoutFunction.apply

    def forward(self, x):
        """
        Теоретическая реализация предполагает расчет на
            - training: y = f(WX) * m
            - testing: y = (1 - p) * f(WX)
        m - маскирующиая матрица из {0, 1} для отключения нейрона.
        Элемент принимает 0 с вероятностью p = self.probability
        Для практической реализации поделим оба выражения на (1 - p)
        :param x:
        :return:
        """
        if self.training:
            # сразу нормируем выход на обучении, поскольку нейрон в среднем принимает (1 - p) информации
            return self.dropout(x, self.probability) / (1 - self.probability)
        else:
            # отключаем Dropout на тесте
            return self.dropout(x, 0)
