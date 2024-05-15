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
        name = legend_name(optimizer.name, plot_filter)
        if name is None:
            continue

        df = pd.concat(
            [latest_metrics(seed_dir) for seed_dir in optimizer.glob("seed=*")]
        )
        metrics.append({"name": name, "metrics": df})

    metrics.sort(key=lambda x: x["name"])
    return metrics


# fmt: off
PLOT_FILTER = {
    "RFD(SqEx)": {
        "includes": ["RFD", "SquaredExponential"],
        "excludes": ["b_size_inv"],
    },
    "RFD(RatQ(beta=1))": {
        "includes": ["RFD", "RationalQuadratic", "beta=1"],
    },
    "Adam(untuned)": {
        "includes": ["Adam", "lr=0.001"],
    },
    "SGD(untuned)": {
        "includes": ["SGD", "lr=0.001"],
    },
    "A-RFD": {
        "includes": ["SGD", "lr=14.2"],
    },
}
# fmt: on
def plot_filter(wanted="all"):
    if wanted == "all":
        return PLOT_FILTER
    
    return {key: val for key, val in PLOT_FILTER.items() if key in wanted}


def plot_metric(ax, metrics, metric, epoch=False, idx=None):
    plot_kwargs= {}
    if idx is not None:
        plot_kwargs["color"] = f"C{idx}"
    
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
    (line,) = ax.plot(agg_df.index, agg_df["mean"], label=metrics["name"], **plot_kwargs)
    ax.fill_between(agg_df.index, agg_df["q0.1"], agg_df["q0.9"], alpha=0.3)

    # seed_grouped_df = reduced_df.groupby("seed")
    # for seed in seed_grouped_df.groups.keys():
    #     seed_df = seed_grouped_df.get_group(seed)
    #     ax.plot(seed_df["epoch"], seed_df[metric], alpha=0.2, color=line.get_color())


def plot_validation_loss(ax, metrics, **kwargs):
    plot_metric(ax, metrics, "val/loss", epoch=True, **kwargs)
    ax.legend()
    ax.set_title("Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")


def plot_accuracy(ax, metrics, **kwargs):
    plot_metric(ax, metrics, "val/accuracy", epoch=True, **kwargs)
    ax.legend()
    ax.set_title("Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")


def plot_initial_learning_rate(ax, metrics, **kwargs):
    df = metrics["metrics"]
    plot_learning_rate(ax, {"name": metrics["name"], "metrics": df[df["step"] < 100]}, **kwargs)


def plot_learning_rate(ax, metrics, **kwargs):
    plot_metric(ax, metrics, "learning_rate", epoch=False, **kwargs)
    ax.legend()
    ax.set_title("Learning Rate")
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")


def plot_step_size(ax, metrics, **kwargs):
    plot_metric(ax, metrics, "step_size", epoch=False, **kwargs)
    ax.legend()
    ax.set_title("Step Size")
    ax.set_xlabel("Step")
    ax.set_yscale("log")

def plot_dot_grad_param(ax, metrics, **kwargs):
    plot_metric(ax, metrics, "dot(grad,param)", epoch=False, **kwargs)
    ax.set_title("dot(grad,param)")
    ax.set_xlabel("Step")
    ax.legend()

def plot_initial_dot_grad_param(ax, metrics, **kwargs):
    df = metrics["metrics"]
    plot_dot_grad_param(ax, {"name": metrics["name"], "metrics": df[df["step"] < 100]}, **kwargs)

def plot_param_norm(ax, metrics, **kwargs):
    plot_metric(ax, metrics, "param_size", epoch=False, **kwargs)
    ax.set_title("Parameter Norm")
    ax.set_xlabel("Step")
    ax.set_yscale("log")
    ax.legend()

def plot_performance(problem):
    problem_dir = Path(f"logs/{problem}")
    metrics = extract_metrics(problem_dir, PLOT_FILTER)
    (fig, axs) = plt.subplots(2, 2, figsize=(10, 6))
    for (idx, item) in enumerate(metrics):
        plot_validation_loss(axs[0, 0], item, idx=idx)
        plot_accuracy(axs[0, 1], item, idx=idx)
        plot_initial_learning_rate(axs[1, 0], item, idx=idx)
        if not item["name"] == "A-RFD":
            plot_step_size(axs[1, 1], item, idx=idx)

    fig.tight_layout()
    plt.savefig(f"plot/{problem}_performance.pdf")


def plot_step_behavior(problem):
    problem_dir = Path(f"logs/{problem}")
    wanted = ["RFD(SqEx)", "RFD(RatQ(beta=1))", "A-RFD"]
    metrics = extract_metrics(problem_dir, plot_filter(wanted))
    (fig, axs) = plt.subplots(3, 2, figsize=(7, 12))
    for (idx, item) in enumerate(metrics):
        plot_learning_rate(axs[0, 0], item, idx=idx)
        plot_initial_learning_rate(axs[1, 0], item, idx=idx)
        plot_step_size(axs[2, 0], item, idx=idx)
        plot_dot_grad_param(axs[0, 1], item, idx=idx)
        plot_initial_dot_grad_param(axs[1, 1], item, idx=idx)
        # if not item["name"] == "A-RFD":
        plot_param_norm(axs[2, 1], item, idx=idx)

    metrics = extract_metrics(problem_dir, plot_filter(["Adam(untuned)", "SGD(untuned)"]))
    for (idx, item) in enumerate(metrics, start=3):
        plot_step_size(axs[2, 0], item, idx=idx)
        plot_dot_grad_param(axs[0, 1], item, idx=idx)
        plot_initial_dot_grad_param(axs[1, 1], item, idx=idx)
        plot_param_norm(axs[2, 1], item, idx=idx)
    
    fig.tight_layout()
    plt.savefig(f"plot/{problem}_steps.pdf")

def main():
    for problem in PROBLEMS:
        # plot_performance(problem)
        plot_step_behavior(problem)


if __name__ == "__main__":
    main()
