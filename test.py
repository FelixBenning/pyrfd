import os
import argparse
from mnistSimpleCNN.train import run
from pyrfd import RFD

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--trials", default=15, type=int)
    p.add_argument("--epochs", default=150, type=int)
    p.add_argument("--kernel_size", default=5, type=int)
    p.add_argument("--gpu", default=0, type=int)
    p.add_argument("--logdir", default="temp")
    args = p.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for i in range(args.trials):
        run(
            RFD,
            p_seed=args.seed + i,
            p_epochs=args.epochs,
            p_kernel_size=args.kernel_size,
            p_logdir=args.logdir,
        )