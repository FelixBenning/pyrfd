"""
Module for sampling the loss at different batch sizes for covariance estimation.
"""

import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


def _budget(bsize_counts):
    return sum(bsize * count for bsize, count in bsize_counts.items())


class CachedSamples:
    """Abstraction for samples collected so far

    Acts as a context manager when adding new samples to allows for KeyboardInterrupt
    while still saving all generated samples so far. Saving them to file
    """

    def __init__(self, filename=None):
        self.filename = filename
        if filename is None:
            self.records = []
        else:
            try:
                self.records = pd.read_csv(filename).to_dict("records")
            except FileNotFoundError:
                self.records = []

    def as_dataframe(self):
        """Returns a copy (not reference!) of the current samples in the form of a dataframe"""
        return pd.DataFrame.from_records(self.records)

    def __len__(self):
        return len(self.records)

    def __enter__(self):
        return self.records

    def __exit__(self, excep_type, excep_val, exc_traceback):
        if self.filename is not None and len(self.records) > 0:
            Path(self.filename).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.records).to_csv(self.filename, index=False)


class IsotropicSampler:
    """Sampling the loss function under the isotropy assumption (i.e. randomly
    samples inputs and does not treat them differently)"""

    def __init__(self, model_factory, loss, data) -> None:
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

    def sample(
        self,
        bsize_counts: pd.Series,
        append_to: CachedSamples = CachedSamples(),
    ):
        """sample the batchsize counts and append them to the cached samples
        (which are used as a context manager to allow for KeyboardInterupt)"""
        budget = _budget(bsize_counts)
        with append_to as records:
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
