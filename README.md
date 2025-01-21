# pyrfd

[![PyPI version](https://badge.fury.io/py/pyrfd.svg)](https://badge.fury.io/py/pyrfd)
[![codecov](https://codecov.io/gh/FelixBenning/pyrfd/graph/badge.svg?token=DaSPgLnZRc)](https://codecov.io/gh/FelixBenning/pyrfd)

Pytorch implementation of RFD (see [arXiv](https://arxiv.org/abs/2305.01377))

## Covariance model

Provides an implementation of the `SquaredExponential` covariance model
with an `auto_fit` function, which requires only
1. A `model_factory` which returns the same but **randomly initialized** model every time it is called
2. A `loss` function e.g. `torch.nn.functional.nll_loss`
which accepts a prediction and a true value
3. data, which can be passed to `torch.utils.DataLoader` with different batch size parameters such
    that it returns `(x,y)` tuples when iterated on
4. a `csv` filename which acts as the cache for the covariance model ofthis
unique (model, data, loss) combination.

## Implementation of RFD

Such a covariance model can then be passed to `RFD` which implements the
pytorch optimizer interface. The end result can be used like `torch.optim.Adam`

## Example usage

```python
from benchmaking.classification.mnist.models.cnn3 import CNN3

import torch
import torchvision as tv

from pyrfd import RFD, SquaredExponential

cov_model = SquaredExponential()
cov_model.auto_fit(
    model_factory=CNN3,
    loss=torch.nn.functional.nll_loss,
    data= tv.datasets.MNIST(
        root="mnistSimpleCNN/data",
        train=True,
        transform=tv.transforms.ToTensor()
    ),
    cache="cache/CNN3_mnist.csv",
    # should be unique for (models, data, loss)
)
rfd = RFD(
    CNN3().parameters(),
    covariance_model=cov_model
)
```

## How to cite


```bibtex
@inproceedings{benningRandomFunctionDescent2024,
  title = {Random {{Function Descent}}},
  booktitle = {Advances in {{Neural Information Processing Systems}}},
  author = {Benning, Felix and D{\"o}ring, Leif},
  year = {2024},
  month = dec,
  volume = {37},
  primaryclass = {cs, math, stat},
  publisher = {Curran Associates, Inc.},
  address = {Vancouver, Canada},
}
```

