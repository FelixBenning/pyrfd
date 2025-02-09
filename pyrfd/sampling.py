"""
Module for sampling the loss at different batch sizes for covariance estimation.
"""

from __future__ import annotations
from abc import abstractmethod

from dataclasses import dataclass, asdict
import time
from pathlib import Path
from logging import warning
from typing import Iterable

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from pyrfd import batchsize


def budget_use(bsize_counts):
    """calculate the used budget from b_size counts"""
    return sum(bsize * count for bsize, count in bsize_counts.items())


class SampleCache:
    """Base Class for Abstraction for Samples collected so far

    Acts as a context manager when adding new samples to allows for
    KeyboardInterrupt while still saving all generated samples so far.

    `self._records` should be a list of records (dictionaries such that
    `pd.DataFrame.from_records` works) with all the current samples.

    Subclasses need to implement
    `__init__` which populates self._records and `__exit__`, which saves
    `self._records`
    """

    __slots__ = ["_records"]

    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError

    def as_dataframe(self):
        """Returns a copy (not reference!) of the current samples in the form of a dataframe"""
        return pd.DataFrame.from_records(self._records)  # pylint: disable=no-member

    def __len__(self):
        return len(self._records)  # pylint: disable=no-member

    def __enter__(self):
        return self._records  # pylint: disable=no-member

    @abstractmethod
    def __exit__(self, excep_type, excep_val, exc_traceback):
        raise NotImplementedError


class CSVSampleCache(SampleCache):
    """Abstraction for samples collected so far

    Acts as a context manager when adding new samples to allows for KeyboardInterrupt
    while still saving all generated samples so far. Saving them to CSV
    """

    __slots__ = ["filename"]

    def __init__(self, filename=None):
        self.filename = filename
        if filename is None:
            warning(
                "Without a cache it is necessary to re-fit the covariance model"
                + "every time. Please provide the path to a viable cache location"
            )
            self._records = []
        else:
            try:
                self._records = pd.read_csv(filename).to_dict("records")
            except FileNotFoundError:
                self._records = []

    def __exit__(self, excep_type, excep_val, exc_traceback):
        if self.filename is not None and len(self._records) > 0:
            Path(self.filename).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self._records).to_csv(self.filename, index=False)

@dataclass
class Sample:
    radius: torch.Tensor
    loss: torch.Tensor
    dx: torch.Tensor
    perp_norm: torch.Tensor

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}

class Plist:
    def __init__(self, params: Iterable[nn.Parameter]):
        object.__setattr__(self, "params", params)

    def __getattr__(self, name):
        return [getattr(param, name) for param in self.params]

    def __setattr__(self, name, value: Iterable[nn.Parameter]):
        for param, v in zip(self.params, value):
            setattr(param, name, v)

    def __iter__(self):
        return iter(self.params)

def norm(params: Iterable[nn.Parameter]):
    return torch.cat([param.detach().flatten() for param in params]).norm(p=2)

def dot(params1: Iterable[nn.Parameter], params2: Iterable[nn.Parameter]):
    return sum(
        torch.dot(param1.flatten(), param2.flatten())
        for param1, param2 in zip(params1, params2)
    )

@torch.no_grad()
def normalize(model: nn.Module, radius=torch.tensor(1.0)):
    """normalize the parameters of the model to lenght radius (default=1)"""
    if radius is None:
        return model
    param_norm = norm(model.parameters())
    for param in model.parameters():
        param *= radius / param_norm
    return model

    # assert torch.isclose(norm(model.parameters()), radius) # debug

@torch.no_grad()
def decomp_grad(params: Plist):
    """decompose gradient of params into orthogonal components
    returns dx (the component of the gradient in the direction of the parameters)
    stores the orthogonal component of the gradient in the grad_perp attribute of the parameters
    """
    param_norm = norm(params)
    for param in params:
        param.hat = param / param_norm

    dx = dot(params.grad, params.hat)
    for param in params:
        param.grad_perp = param.grad - dx * param.hat

    return dx




class IsotropicSampler:
    """Sampling the loss function under the isotropy assumption (i.e. randomly
    samples inputs and does not treat them differently)"""

    def __init__(
        self,
        model_factory,
        loss,
        data,
        cache: SampleCache | str | None = None,
        seed=None,
    ) -> None:
        self.model_factory = model_factory
        self.data = data
        self.loss = loss
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

        if isinstance(cache, str):
            cache = CSVSampleCache(cache)
        self.cache = cache
        self._dims = sum(
            p.numel() for p in model_factory().parameters() if p.requires_grad
        )

    def snapshot_as_dataframe(self):
        """Returns a copy of the current samples in the form of a dataframe"""
        if self.cache is None:
            return pd.DataFrame()
        return self.cache.as_dataframe()

    def __len__(self):
        if self.cache is None:
            return 0
        return len(self.cache)

    @property
    def dims(self):
        """Returns the dimension of the model parameters of the model factory"""
        return self._dims

    @property
    def bsize_counts(self):
        """Returns the counts of batch sizes in the cache"""
        if self.cache is None:
            return pd.Series()
        return self.cache.as_dataframe().get("batchsize", pd.Series()).value_counts()

    @property
    def sample_cost(self):
        """calculate the cost of sampling the batchsize counts"""
        return budget_use(self.bsize_counts)

    def loader(self, batch_size, num_samples):
        sampler = RandomSampler(
            data_source=self.data,
            replacement=True,
            generator=self.generator,
            num_samples=batch_size * num_samples,
        )
        return DataLoader(self.data, batch_size=batch_size, sampler=sampler)
    
    @torch.enable_grad()
    def loss_and_grad(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor):
        """ compute the loss and gradient"""
        prediction = model(x)
        loss = self.loss(prediction, y)
        loss.backward()
        return loss

    def loss_and_grad_sample(self, x, y, radius=None):
        model = normalize(self.model_factory(), radius)
        model.zero_grad()
        loss = self.loss_and_grad(model, x, y)

        params = Plist(list(model.parameters()))
        dx = decomp_grad(params)
        perp_norm = norm(params.grad_perp)
        # print(f"{dx=}, {perp_norm=}, grad_norm={norm(params.grad)}")
        # assert torch.isclose(norm(params.grad), torch.tensor((perp_norm, dx)).norm())
        # assert torch.isclose(dot(params, params.grad_perp), torch.tensor(0.))
        return Sample(norm(params), loss, dx, perp_norm)

    def sample(self, bsize_counts: pd.Series):
        """sample the batchsize counts and append them to the cached samples
        (which are used as a context manager to allow for KeyboardInterupt)"""
        if self.cache is None:
            self.cache = CSVSampleCache()

        budget = budget_use(bsize_counts)
        pbar = tqdm(
            total=budget,
            unit="samples",
            desc="Loss/gradient sampling",
            position=1,
            leave=False,
        )
        with self.cache as records, pbar as progress:
            for b_size, count in bsize_counts.items():
                data_loader = self.loader(b_size, count)
                for x,y in data_loader:
                    records.append(dict(
                        time= time.time(),
                        batchsize= b_size,
                        **self.loss_and_grad_sample(x, y).dict(),
                    ))
                    progress.update(b_size)
                progress.set_description(f"Loss/gradient sampling (batchsize={b_size})") 
        return budget
