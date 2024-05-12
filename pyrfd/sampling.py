"""
Module for sampling the loss at different batch sizes for covariance estimation.
"""

from __future__ import annotations
from abc import abstractmethod

import time
from pathlib import Path
from logging import warning

import pandas as pd
import torch
from tqdm import tqdm


def budget_use(bsize_counts):
    """calculate the used budget from b_size counts"""
    return sum(bsize * count for bsize, count in bsize_counts.items())

class SampleCache:
    """ Base Class for Abstraction for Samples collected so far

    Acts as a context manager when adding new samples to allows for
    KeyboardInterrupt while still saving all generated samples so far.

    `self._records` should be a list of records (dictionaries such that
    `pd.DataFrame.from_records` works) with all the current samples.

    Subclasses need to implement
    `__init__` which populates self._records and `__exit__`, which saves
    `self._records`
    """

    __slots__ = [ "_records" ]

    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError

    def as_dataframe(self):
        """Returns a copy (not reference!) of the current samples in the form of a dataframe"""
        return pd.DataFrame.from_records(self._records) # pylint: disable=no-member

    def __len__(self):
        return len(self._records) # pylint: disable=no-member

    def __enter__(self):
        return self._records # pylint: disable=no-member

    @abstractmethod
    def __exit__(self, excep_type, excep_val, exc_traceback):
        raise NotImplementedError

class CSVSampleCache(SampleCache):
    """Abstraction for samples collected so far

    Acts as a context manager when adding new samples to allows for KeyboardInterrupt
    while still saving all generated samples so far. Saving them to CSV 
    """

    __slots__ = [ "filename" ]

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


class IsotropicSampler:
    """Sampling the loss function under the isotropy assumption (i.e. randomly
    samples inputs and does not treat them differently)"""

    def __init__(self, model_factory, loss, data, cache: SampleCache | str | None=None) -> None:
        if isinstance(cache, str):
            cache = CSVSampleCache(cache)
        self.cache = cache
        self._dims = sum(
            p.numel() for p in model_factory().parameters() if p.requires_grad
        )

        def loader(b_size):
            return torch.utils.data.DataLoader(data, batch_size=b_size, shuffle=True)

        def loss_sample(input_x, target_y):
            model = model_factory()
            # this is a weird way to set the gradients to zero but pytorch...
            torch.optim.SGD(model.parameters()).zero_grad()
            with torch.enable_grad():
                prediction = model(input_x)
                sample_loss = loss(prediction, target_y)
                sample_loss.backward()

            with torch.no_grad():
                grads = [
                    param.grad.detach().flatten()
                    for param in model.parameters()
                    if param.grad is not None
                ]
                grad_norm = torch.cat(grads).norm()
            return sample_loss.item(), grad_norm.item()

        self.loader = loader
        self.loss_sample = loss_sample

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
        """ Returns the dimension of the model parameters of the model factory """
        return self._dims


    @property
    def bsize_counts(self):
        """Returns the counts of batch sizes in the cache"""
        if self.cache is None:
            return pd.Series()
        return self.cache.as_dataframe()["batchsize"].value_counts()

    @property
    def sample_cost(self):
        """calculate the cost of sampling the batchsize counts"""
        return budget_use(self.bsize_counts)

    def sample(self, bsize_counts: pd.Series):
        """sample the batchsize counts and append them to the cached samples
        (which are used as a context manager to allow for KeyboardInterupt)"""
        if self.cache is None:
            self.cache = CSVSampleCache()

        budget = budget_use(bsize_counts)
        with self.cache as records:
            with tqdm(
                total=budget,
                unit="samples",
                desc="Loss/gradient sampling",
                position=1,
                leave=False,
            ) as progress:
                for b_size, count in bsize_counts.items():
                    self._sample_batchloss(
                        b_size, count, append_to=records, progress=progress
                    )
        return budget

    def _sample_batchloss(self, b_size, count, append_to, progress: tqdm | None = None):
        data_loader = self.loader(b_size)
        data_iter = iter(data_loader)
        for _ in range(count):
            try:
                x, y = next(data_iter)
            except StopIteration:
                # need to reinitialized loader
                data_iter = iter(data_loader)
                x, y = next(data_iter)
            loss, g_norm = self.loss_sample(x, y)
            append_to.append(
                {
                    "loss": loss,
                    "grad_norm": g_norm,
                    "batchsize": b_size,
                    "time": time.time(),
                }
            )
            if progress:
                progress.update(b_size)
                progress.set_description(f"Loss/gradient sampling (batchsize={b_size})")
        return append_to
