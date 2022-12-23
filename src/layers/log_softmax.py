from typing import Any

import torch
import torch.nn as nn


class LogSoftmaxFunction(torch.autograd.Function):
    """
    Функция для реализации прямого и обратного прохода
    LogSoftmax слоя нейронной сети
    """
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Softmax(X) = exp(x_i) / sum(exp(x_j)), где i = 1...K (число классов)
        Для избежания переполнения в экспоненте используем Log-Sum-Exp trick.
        Можно использовать готовую функцию torch.logsumexp или самим ее написать:
            x_max = torch.max(x)
            s = torch.exp(x - x_max).sum()
            lse = torch.log(s)
        Получим softmax(X) = exp(x_i) / sum(exp(x_j)) = exp(x_i - lse(x))
            -> log(softmax(x)) = log(exp(x_i - lse)) = x_i - lse
        :param ctx: контекст для сохранения производных при прямом обходе
        :param args: аргументы в порядке [X,]
        :param kwargs:
        :return:
        """
        x = args[0]
        lse = torch.logsumexp(x, dim=-1).view(-1, 1)
        log_softmax = x - lse
        ctx.save_for_backward(log_softmax)
        return log_softmax


    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """
        Обратное распростанение градиента для LogSoftmax(X)
        Для нашей конкретной функции потерь все вычисления можно реализовать в матричном виде.
        Поэтому будем предполагать, что аргумент grad_outputs[0] - это матрица, у которой
        в каждой строке только одно ненулевое значение (не обязательно единица)

        dL/dX = dL/dz * dz/dX = dL/dz * (1 - exp(logsoftmax(x)))
            z = x - lse(x)
            dz/dx = 1 - exp(x_i) / sum(exp(x_j)) = 1 - softmax(x) = 1 - exp(logsoftmax(x))

        :param ctx: контекст с сохраненными значениями при прямом проходе
        :param grad_outputs: приходящий градиент
        :return: градиенты по dX
        """
        log_softmax = ctx.saved_tensors[0]
        grad_output = grad_outputs[0]
        result = grad_output - torch.sum(grad_output, dim=-1).view(-1, 1) * torch.exp(log_softmax)
        return result


class LogSoftmax(nn.Module):
    """
    Реализация LogSoftmax слоя нейронной сети.
    Для переопределения метода backward требуется создать свою функцию
    и вызывать ее методы.
    """
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmaxFunction.apply

    def forward(self, x):
        return self.log_softmax(x)
