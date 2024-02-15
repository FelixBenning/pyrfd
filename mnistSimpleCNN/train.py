# imports -------------------------------------------------------------------------#
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from .ema import EMA
from .datasets import MnistDataset, DIR
from .transforms import RandomRotation
from .models.modelM3 import ModelM3
from .models.modelM5 import ModelM5
from .models.modelM7 import ModelM7


def run(opt=optim.Adam, p_seed=0, p_epochs=10, p_kernel_size=5, p_logdir="temp"):
    # random number generator seed ------------------------------------------------#
    SEED = p_seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    # kernel size of model --------------------------------------------------------#
    KERNEL_SIZE = p_kernel_size

    # number of epochs ------------------------------------------------------------#
    NUM_EPOCHS = p_epochs

    # file names ------------------------------------------------------------------#
    if not os.path.exists(DIR.joinpath("logs/%s" % p_logdir)):
        os.makedirs(DIR.joinpath("logs/%s" % p_logdir))
    OUTPUT_FILE = str(DIR.joinpath("logs/%s/log%03d.out" % (p_logdir, SEED)))
    MODEL_FILE = str(DIR.joinpath("logs/%s/model%03d.pth" % (p_logdir, SEED)))

    # enable GPU usage ------------------------------------------------------------#
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda == False:
        print("WARNING: CPU will be used for training.")
        # exit(0)

    # data augmentation methods ---------------------------------------------------#
    transform = transforms.Compose(
        [
            RandomRotation(20, seed=SEED),
            transforms.RandomAffine(0, translate=(0.2, 0.2)),
        ]
    )

    # data loader -----------------------------------------------------------------#
    train_dataset = MnistDataset(training=True, transform=transform)
    test_dataset = MnistDataset(training=False, transform=None)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=120, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False
    )

    # model selection -------------------------------------------------------------#
    if KERNEL_SIZE == 3:
        model = ModelM3().to(device)
    elif KERNEL_SIZE == 5:
        model = ModelM5().to(device)
    elif KERNEL_SIZE == 7:
        model = ModelM7().to(device)

    summary(model, (1, 28, 28))

    # hyperparameter selection ----------------------------------------------------#
    ema = EMA(model, decay=0.999)
    optimizer = opt(model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # delete result file ----------------------------------------------------------#
    f = open(OUTPUT_FILE, "w")
    f.close()

    # global variables ------------------------------------------------------------#
    g_step = 0
    max_correct = 0

    # training and evaluation loop ------------------------------------------------#
    epoch_pg = tqdm(range(NUM_EPOCHS), desc="Train Epoch", position=0)
    for epoch in epoch_pg:
        # --------------------------------------------------------------------------#
        # train process                                                            #
        # --------------------------------------------------------------------------#
        model.train()
        train_loss = 0
        train_corr = 0
        pbar = tqdm(
            train_loader,
            desc="Batch",
            position=1,
        )
        for data, target in pbar:
            data, target = data.to(device), target.to(device, dtype=torch.int64)
            def loss_closure():
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                train_pred = output.argmax(dim=1, keepdim=True)

                nonlocal train_corr
                train_corr += train_pred.eq(target.view_as(train_pred)).sum().item()
                nonlocal train_loss
                train_loss += F.nll_loss(output, target, reduction="sum").item()

                loss.backward()
                pbar.set_postfix({"loss": loss.item()}) 
                return loss
            optimizer.step(loss_closure)
            g_step += 1
            ema(model, g_step)
        train_loss /= len(train_loader.dataset)
        train_accuracy = 100 * train_corr / len(train_loader.dataset)

        # --------------------------------------------------------------------------#
        # test process                                                             #
        # --------------------------------------------------------------------------#
        model.eval()
        ema.assign(model)
        test_loss = 0
        correct = 0
        total_pred = np.zeros(0)
        total_target = np.zeros(0)
        test_pg = tqdm(test_loader, desc="Testing", position=1, leave=False)
        with torch.no_grad():
            for data, target in test_pg:
                data, target = data.to(device), target.to(device, dtype=torch.int64)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                total_pred = np.append(total_pred, pred.cpu().numpy())
                total_target = np.append(total_target, target.cpu().numpy())
                correct += pred.eq(target.view_as(pred)).sum().item()
            if max_correct < correct:
                torch.save(model.state_dict(), MODEL_FILE)
                max_correct = correct
        ema.resume(model)

        # --------------------------------------------------------------------------#
        # output                                                                   #
        # --------------------------------------------------------------------------#
        test_loss /= len(test_loader.dataset)
        test_accuracy = 100 * correct / len(test_loader.dataset)
        best_test_accuracy = 100 * max_correct / len(test_loader.dataset)
        tqdm.write(
            f"Average loss: {test_loss}, "
            f"Accuracy: {test_accuracy:.2f} ({correct}/{len(test_loader.dataset)})"
        )
        epoch_pg.set_postfix({
            "Accuracy": best_test_accuracy
        })

        f = open(OUTPUT_FILE, "a")
        f.write(
            " %3d %12.6f %9.3f %12.6f %9.3f %9.3f\n"
            % (
                epoch,
                train_loss,
                train_accuracy,
                test_loss,
                test_accuracy,
                best_test_accuracy,
            )
        )
        f.close()

        # --------------------------------------------------------------------------#
        # update learning rate scheduler                                           #
        # --------------------------------------------------------------------------#
        lr_scheduler.step()
