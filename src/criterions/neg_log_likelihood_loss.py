from typing import Any

import torch
import torch.nn as nn


class NLLLossFunction(torch.autograd.Function):
    """
    Функция для реализации прямого и обратного прохода
    функции потерь Negative Log Likelihood Loss нейронной сети
    """
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Логарифм правдоподобия: L(activation, target) = - sum(target[i] * log(activation[i])) / M -> min,
        где target[i] = 1, если объект принадлежит i-ому классу, иначе target[i] = 0.
        Эту запись можно переписать:
        NLLLoss(activation, target) = - sum(log(activation[i]|target[i] = 1)) / M
            activation - выходы линейного слоя после LogSoftmax()
            target - тензор с метками класса (tensor([0, 2, 1, ...]))
            M - число объектов выборки
        :param ctx: контекст для сохранения производных при прямом обходе
        :param args: аргументы в порядке [activation, target]
        :param kwargs:
        :return:
        """
        activation = args[0]
        target = args[1]
        ctx.save_for_backward(activation, target)
        # можно сделать через torch.gather(activation, 1, target.to(int).view(-1, 1))
        # аналог в numpy: np.take_along_axis()
        predicted = activation[range(activation.shape[0]), target]
        return - predicted.mean()

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """
        Обратное распростанение градиента для Negative Log Likelihood Loss
        По логике работы приходящий градиент - это число, но последующие
        слои ожидают градиент в виде матрицы, поэтому нужен тензор с ненулевыми
        значениями в индексах target

        dL/d(activation) = dL/dz * dz/d(activation) = dL/dz * (- 1 / M)
            M - число объектов выборки
        dL/d(target) = None, как как target - метки класса (const)

        :param ctx: контекст с сохраненными значениями при прямом проходе
        :param grad_outputs: приходящий градиент
        :return: градиенты по d(activation), d(target)
        """
        activation, target = ctx.saved_tensors[0], ctx.saved_tensors[1]
        grad_matrix = torch.zeros_like(activation)
        grad_matrix[range(activation.shape[0]), target] = -1
        return grad_outputs[0] * grad_matrix / len(target), None


class NLLLoss(nn.Module):
    """
    Реализация функции потерь Negative Log Likelihood Loss для нейронной сети.
    Для переопределения метода backward требуется создать свою функцию
    и вызывать ее методы.
    """
    def __init__(self):
        super().__init__()
        self.nllloss = NLLLossFunction.apply

    def forward(self, activation, target):
        return self.nllloss(activation, target)
