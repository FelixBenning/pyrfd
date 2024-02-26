# pyrfd

Pytorch implementation of RFD

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
from mnistSimpleCNN.models.modelM3 import ModelM3
# cf. mnistSimpleCNN directory (example model)

import torch
import torchvision as tv

from pyrfd import RFD, SquaredExponential

cov_model = SquaredExponential()
cov_model.auto_fit(
    model_factory=ModelM3,
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
    ModelM3().parameters(),
    covariance_model=cov_model
)
```