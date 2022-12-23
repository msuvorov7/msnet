from typing import Optional, Callable

import torch


class SGD(torch.optim.Optimizer):
    """
    Реализация классического метода стохастического градиентного спуска для нейронной сети.
    """
    def __init__(self, params, lr: float = 1e-3):
        """
        Метод градиентного спуска
        :param params: параметры модели для пересчета градиента
        :param lr: learning rate
        """
        opt_params = {
            'lr': lr,
        }
        super().__init__(params, opt_params)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = ...) -> Optional[float]:
        """
        Расчет итеративного обновления весов по формуле:
        w_(t+1) = w_(t) - lr * grad(L(w_(t))),
            w_(t+1) - значения весов на следующей итерации (t+1)
            w_(t) - значения весов на итерации t
            lr - learning rate
            grad(*) - градиент от *
            L(w_(t)) - функция потерь от w_(t)
        :param closure:
        :return:
        """
        for group in self.param_groups:
            lr = group['lr']
            for param in group['params']:
                if param.grad is not None:
                    param.data += (- lr * param.grad)
        return
