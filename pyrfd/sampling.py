import pandas as pd
import torch
import tqdm
import time

class CachedSamples:
    def __init__(self, filename=None):
        self.filename = filename
        if filename is None:
            self.records = []
        else:
            self.records = pd.read_csv(filename).to_records()

    def __enter__(self):
        return self.records
        
    def __exit__(self, excep_type, excep_val, exc_traceback):
        if self.filename is not None:
            pd.DataFrame(self.records).to_csv(self.filename)


class IsotropicSampler:
    def __init__(self, model_factory, loss, data) -> None:
        def loader(b_size):
            return torch.utils.data.DataLoader(data, batch_size=b_size, shuffle=True)

        def loss_sample(input, target):
            model = model_factory()
            # this is a weird way to set the gradients to zero but pytorch...
            torch.optim.SGD(model.parameters()).zero_grad()
            with torch.enable_grad():
                prediction = model(input)
                sample_loss = loss(prediction, target)
                sample_loss.backward()

            with torch.no_grad():
                grads = [
                    param.grad.detach().flatten()
                    for param in model.parameters()
                    if param.grad is not None
                ]
                grad_norm = torch.cat(grads).norm()
            return sample_loss.item(), grad_norm
        
        self.loader = loader
        self.loss_sample = loss_sample
    
    def sample(self, batchsizes, cachedSamples=CachedSamples()):
        with cachedSamples as records:
            for b_size, count in batchsizes.items():
                self.sample_batchloss(b_size, count, append_to=records)
                    
    def sample_batchloss(self, b_size, count, append_to=[]):
        data_loader = self.loader(b_size)
        data_iter = iter(data_loader)
        for _ in tqdm(range(count), desc=f"Sampling batchsize={b_size}"):
            try:
                x,y = next(data_iter)
            except StopIteration:
                # need to reinitialized loader
                data_iter = iter(data_loader)     
                x,y = next(data_iter)
            loss, g_norm = self.loss_sample(x,y)
            append_to.append(
                {
                    "loss": loss,
                    "grad_norm": g_norm,
                    "batchsize": b_size,
                    "time": time.time(),
                }
            )
        return append_to

