# msnet

Репозиторий для узучения DeepLearning на фреймворке PyTorch.

## Структура
Проект содержит реализации таких слоев нейронной сети как:
- [Linear](./src/layers/linear.py) - линейный (полносвязный) слой
- [ReLU](./src/layers/relu.py) - слой функции активании ReLU
- [Sigmoid](./src/layers/sigmoid.py) - слой функции активании Sigmoid
- [Dropout](./src/layers/dropout.py) - Dropout слой (случайное выбрасывание нейронов)
- [LogSoftmax](./src/layers/log_softmax.py) - LogSoftmax слой (для многоклассовой классификации)

Доступные функции потерь:
- [NLLLoss](./src/criterions/neg_log_likelihood_loss.py) - Negative Log Likelihood Loss
- [CrossEntropyLoss](./src/criterions/cross_entropy_loss.py) - CrossEntropyLoss, представленная как LogSoftmax + NLLLoss

