from telnetlib import DO
import pandas as pd
from pathlib import Path
import re

from matplotlib import legend, rc
import matplotlib.pyplot as plt
from sympy import plot


rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
plt.rcParams.update(
    {
        # 'text.usetex': True,
        "text.latex.preamble": r"\usepackage{amsfonts}"
    }
)

PROBLEMS = ["MNIST_CNN7_b=1024"]


def latest_metrics(seed_dir):
    seed_nr = int(seed_dir.name.split("=")[1])
    all_versions = list(seed_dir.glob("version_*"))
    if len(all_versions) == 0:
        return pd.DataFrame()

    last_version = max(
        seed_dir.glob("version_*"), key=lambda x: int(re.search(r"\d+", x.name).group())
    )
    metrics = last_version.joinpath("metrics.csv")
    if not metrics.exists():
        return pd.DataFrame()

    df = pd.read_csv(metrics)
    df["seed"] = seed_nr
    return df


def legend_name(hyper_params, plot_filter):
    for name, reqs in plot_filter.items():
        if "includes" in reqs:
            if not all(x in hyper_params for x in reqs["includes"]):
                continue

        if "excludes" in reqs:
            if not all(x not in hyper_params for x in reqs["excludes"]):
                continue
        
        return name

    return None


def extract_metrics(problem_dir, plot_filter):
    metrics = []
    for optimizer in problem_dir.glob("*"):
        optimizer_name = optimizer.name.split("(")[0]
        if not optimizer_name in plot_filter:
            continue

        start = optimizer.name.find("(")
        stop = optimizer.name.rfind(")")
        hyper_params = optimizer.name[start + 1 : stop]

        name = legend_name(hyper_params, plot_filter[optimizer_name])
        if name is None:
            continue

        df = pd.concat(
            [latest_metrics(seed_dir) for seed_dir in optimizer.glob("seed=*")]
        )
        metrics.append({"name": name, "metrics": df})
    return metrics


# fmt: off
PLOT_FILTER = {
    "RFD": {
        "RFD(SqEx)": {
            "includes": ["SquaredExponential"],
            "excludes": ["b_size_inv"],
        },
        "RFD(RatQ(beta=1))": {
            "includes": ["RationalQuadratic", "beta=1"],
        },
    },
    "Adam": {
        "Adam(untuned)": {"includes": ["lr=0.001"]}
    },
    "SGD": {
        "SGD(untuned)": {
            "includes": ["lr=0.001"],
        },
        # "A-RFD": {"includes": ["lr=14.2"]},
    },
}
# fmt: on


def plot_metric(ax, metrics, metric, epoch=False):
    time = "epoch" if epoch else "step"
    df = metrics["metrics"]
    reduced_df = df[[time, metric, "seed"]].dropna()

    # sanity check (epoch and seed should be unique together):
    reduced_df.set_index([time, "seed"], verify_integrity=True)

    # fmt: off
    agg_df = reduced_df.groupby(time).agg({metric: [
        "mean",
        pd.NamedAgg("q0.1", lambda x: x.quantile(0.1)),
        pd.NamedAgg("q0.9", lambda x: x.quantile(0.9)),
    ]})[metric]
    # fmt: on
    (line,) = ax.plot(agg_df.index, agg_df["mean"], label=metrics["name"])
    ax.fill_between(agg_df.index, agg_df["q0.1"], agg_df["q0.9"], alpha=0.3)

    # seed_grouped_df = reduced_df.groupby("seed")
    # for seed in seed_grouped_df.groups.keys():
    #     seed_df = seed_grouped_df.get_group(seed)
    #     ax.plot(seed_df["epoch"], seed_df[metric], alpha=0.2, color=line.get_color())


def plot_validation_loss(ax, metrics):
    plot_metric(ax, metrics, "val/loss", epoch=True)
    ax.legend()
    ax.set_title("Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")

def plot_accuracy(ax, metrics):
    plot_metric(ax, metrics, "val/accuracy", epoch=True)
    ax.legend()
    ax.set_title("Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")

def plot_learning_rate(ax, metrics):
    plot_metric(ax, metrics, "learning_rate", epoch=False)
    ax.legend()
    ax.set_title("Learning Rate")
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")

def plot_step_size(ax, metrics):
    plot_metric(ax, metrics, "step_size", epoch=False)
    ax.legend()
    ax.set_title("Step Size")
    ax.set_xlabel("Step")


def plot_performance(problem):
    problem_dir = Path(f"logs/{problem}")
    metrics = extract_metrics(problem_dir, PLOT_FILTER)
    (fig, axs) = plt.subplots(2, 2, figsize=(8, 6))
    for item in metrics:
        plot_validation_loss(axs[0, 0], item)
        plot_accuracy(axs[0, 1], item)
        plot_learning_rate(axs[1, 0], item)
        plot_step_size(axs[1, 1], item)



    fig.tight_layout()
    plt.savefig(f"plot/{problem}.pdf")


def main():
    for problem in PROBLEMS:
        plot_performance(problem)


if __name__ == "__main__":
    main()
