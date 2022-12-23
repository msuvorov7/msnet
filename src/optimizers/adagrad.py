from typing import Optional, Callable

import torch


class AdaGrad(torch.optim.Optimizer):
    """
    Реализация метода AdaGrad для нейронной сети.
    """
    def __init__(self, params, lr: float = 1e-2, eps: float = 1e-6):
        """
        Метод AdaGrad
        :param params: параметры модели для пересчета градиента
        :param lr: learning rate
        :param eps: защита от деления на 0
        """
        # поскольку params - это генератор, его нужно сохранить для обхода в нескольких вызовах
        model_params = list(params)
        opt_params = {
            'lr': lr,
            'eps': eps,
        }
        self.v = [torch.zeros_like(param) for param in model_params]
        super().__init__(model_params, opt_params)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        """
        Расчет итеративного обновления весов по формуле:
        v_(t+1) = v_(t) + grad(L(w_(t)))**2
        w_(t+1) = w_(t) - lr * grad(L(w_(t))) / sqrt(v_(t+1) + eps),
            w_(t+1) - значения весов на следующей итерации (t+1)
            w_(t) - значения весов на итерации t
            v_(t+1) - нормировка темпа сходимости на шаге (t+1)
            v_(0) - матрица из нулей размерности grad(L(w_(t)))
            lr - learning rate
            eps - малая константа для защиты от деления на нуль
            grad(*) - градиент от *
            L(w_(t)) - функция потерь от w_(t)
        :param closure:
        :return:
        """
        for group in self.param_groups:
            lr = group['lr']
            eps = group['eps']
            for i, param in enumerate(group['params']):
                if param.grad is not None:
                    self.v[i] += torch.square(param.grad)
                    param.data += (- lr * param.grad / torch.sqrt(self.v[i] + eps))
        return
