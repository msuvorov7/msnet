from typing import Any

import torch
import torch.nn as nn


class FocalLossFunction(torch.autograd.Function):
    """
    Функция для реализации прямого и обратного прохода
    функции потерь Focal Loss нейронной сети
    """
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Формула: L(activation, target) = - (1 - activation)^(gamma) * log(activation) -> min,
        где target[i] = 1, если объект принадлежит i-ому классу, иначе target[i] = 0.
            activation - выходы линейного слоя после Softmax()
            target - тензор с метками класса (tensor([0, 2, 1, ...]))
            gamma - степень "прижатия" (при росте gamma функция опускается быстрее при насыщении)
            M - число объектов выборки
        :param ctx: контекст для сохранения производных при прямом обходе
        :param args: аргументы в порядке [activation, target, gamma]
        :param kwargs:
        :return:
        """
        activation = args[0]
        target = args[1]
        gamma = args[2]

        ctx.save_for_backward(activation, target, torch.tensor(gamma, requires_grad=False))
        target_mask = (range(activation.shape[0]), target)

        predicted = ((1 - activation[target_mask]) ** gamma) * torch.log(activation)[target_mask]

        return - predicted.mean()

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """
        Обратное распростанение градиента для Focal Loss
        По логике работы приходящий градиент - это число, но последующие
        слои ожидают градиент в виде матрицы, поэтому нужен тензор с ненулевыми
        значениями в индексах target

        dL/d(activation) = dL/dz * dz/d(activation) =
            = dL/dz * (gamma * (1 - activation)^(gamma - 1) * log(activation) - ((1 - activation)^(gamma)) / activation)
            M - число объектов выборки
        dL/d(target) = None, как как target - метки класса (const)
        dL/d(gamma) = None, как как gamma - const

        :param ctx: контекст с сохраненными значениями при прямом проходе
        :param grad_outputs: приходящий градиент
        :return: градиенты по d(activation), d(target), d(gamma)
        """
        activation = ctx.saved_tensors[0]
        target = ctx.saved_tensors[1]
        gamma = ctx.saved_tensors[2]

        grad_matrix = torch.zeros_like(activation)
        target_mask = (range(activation.shape[0]), target)

        grad_matrix[target_mask] = (
                gamma * ((1 - activation[target_mask]) ** (gamma - 1)) * torch.log(activation[target_mask]) -
                ((1 - activation[target_mask]) ** gamma) / activation[target_mask]
        )

        return grad_outputs[0] * grad_matrix / len(target), None, None


class FocalLoss(nn.Module):
    """
    Реализация FocalLoss слоя нейронной сети.
    Для переопределения метода backward требуется создать свою функцию
    и вызывать ее методы.

    Операция SoftMax применяется внутри forward.
    """
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.softmax = nn.Softmax(dim=-1)
        self.focal_loss = FocalLossFunction.apply

    def forward(self, activation, target):
        """
        Прямой проход Focal Loss
        :param activation: выход после Linear
        :param target: метки классов
        :return:
        """
        return self.focal_loss(self.softmax(activation), target, self.gamma)
