from typing import Optional, Callable

import torch


class RMSprop(torch.optim.Optimizer):
    """
    Реализация метода RMSprop для нейронной сети.
    Метод представляет собой доработку AdaGrad в виде добавления скользящего среднего
    по параметру v
    """
    def __init__(self, params, lr: float = 1e-2, alpha: float = 0.8, eps: float = 1e-6):
        """
        Метод RMSprop: AdaGrad + Moving Average
        :param params: параметры модели для пересчета градиента
        :param lr: learning rate
        :param alpha: коэффициент для скользящего среднего
        :param eps: защита от деления на 0
        """
        assert (alpha > 0) and (alpha < 1.0)
        # поскольку params - это генератор, его нужно сохранить для обхода в нескольких вызовах
        model_params = list(params)
        opt_params = {
            'lr': lr,
            'alpha': alpha,
            'eps': eps,
        }
        self.v = [torch.zeros_like(param) for param in model_params]
        super().__init__(model_params, opt_params)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        """
        Расчет итеративного обновления весов по формуле:
        v_(t+1) = alpha * v_(t) + (1 - alpha) * grad(L(w_(t)))**2
        w_(t+1) = w_(t) - lr * grad(L(w_(t))) / sqrt(v_(t+1) + eps),
            w_(t+1) - значения весов на следующей итерации (t+1)
            w_(t) - значения весов на итерации t
            v_(t+1) - нормировка темпа сходимости на шаге (t+1)
            v_(0) - матрица из нулей размерности grad(L(w_(t)))
            lr - learning rate
            alpha - влияние предыдущего значения v
            eps - малая константа для защиты от деления на нуль
            grad(*) - градиент от *
            L(w_(t)) - функция потерь от w_(t)
        :param closure:
        :return:
        """
        for group in self.param_groups:
            lr = group['lr']
            alpha = group['alpha']
            eps = group['eps']
            for i, param in enumerate(group['params']):
                if param.grad is not None:
                    self.v[i] = alpha * self.v[i] + (1 - alpha) * torch.square(param.grad)
                    param.data += (- lr * param.grad / torch.sqrt(self.v[i] + eps))
        return
