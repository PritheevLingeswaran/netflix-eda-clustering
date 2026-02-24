from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt


def plot_bar(series: pd.Series, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax = series.plot(kind="bar")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_hist(series: pd.Series, title: str, bins: int = 30, xlabel: str = "", ylabel: str = "Count") -> None:
    plt.figure()
    plt.hist(series.dropna().values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()
