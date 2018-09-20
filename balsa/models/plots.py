from typing import Callable, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter, FuncFormatter


def convergence_boxplot(targets: pd.DataFrame, results: pd.DataFrame, filter_func: Callable[[pd.Series], pd.Series],
                        adjust_target: bool=True, percentage: bool=True, band: Tuple[float, float]=None,
                        simple_labels: bool=True, ax=None, fp: str=None, title: str=None) -> Axes:
    assert results.columns.equals(targets.columns)

    columns, filters, n = [], [], 0
    for colname in targets:
        filter_ = filter_func(targets[colname])
        filters.append(filter_)
        n = max(n, filter_.sum())
        columns.append(colname)

    unlabelled_zones = np.full([n, len(columns)], np.nan, dtype=np.float64)
    model_sums, target_sums = [], []

    for i, (colname, filter_) in enumerate(zip(columns, filters)):
        m = filter_.sum()

        model_vector = results.loc[filter_, colname]
        target_vector = targets.loc[filter_, colname]

        if adjust_target and target_vector.sum() > 0:
            factor = model_vector.sum() / target_vector.sum()
            target_vector = target_vector * factor

        model_sums.append(model_vector.sum())
        target_sums.append(target_vector.sum())

        err = model_vector - target_vector
        if percentage: err /= target_vector

        unlabelled_zones[:m, i] = err.values

    if not simple_labels:
        columns = [
            f"{c}\n{model_sums[i]} workers\n{int(target_sums[i])} jobs\n{filters[i].sum()} zones"
            for i, c in enumerate(columns)
        ]
    unlabelled_zones = pd.DataFrame(unlabelled_zones, columns=columns)

    with np.errstate(invalid='ignore'):
        ax = unlabelled_zones.plot.box(ax=ax, figsize=[12, 6])
        ax.axhline(0)

        if percentage:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{np.round(x, 2) * 100}%"))
            ax.set_ylabel("Relative error ((Model - Target) / Target)")
        else:
            ax.set_ylabel("Error (Model - Target)")

        xlabel = "Occupation & Employment Status"
        if adjust_target:
            xlabel += "\n(Targets adjusted to match model totals)"
        ax.set_xlabel(xlabel)

        if band is not None:
            lower, upper = band
            ax.axhline(lower, color='black', linewidth=1, alpha=0.5)
            ax.axhline(upper, color='black', linewidth=1, alpha=0.5)

        if title: ax.set_title(title)

        if fp is not None:
            plt.savefig(str(fp))

        return ax


def location_summary(model: pd.DataFrame, target: pd.DataFrame, ensemble_names: pd.Series, title: str='', fp: Path=None,
                     dpi: int=150, district_name: str='Ensemble') -> Axes:
    fig, ax = plt.subplots(1, 3, figsize=[16, 8], gridspec_kw={'width_ratios': [4, 2, 2]})

    model_col = model.reindex(ensemble_names.index, fill_value=0)
    target_col = target.reindex(ensemble_names.index, fill_value=0)

    factor = model_col.sum() / target_col.sum()
    target_col = target_col * factor

    df = pd.DataFrame({'Model': model_col, "Target": target_col})
    df.index = ensemble_names
    df.index.name = district_name
    sub_ax = ax[0]
    df.plot.barh(ax=sub_ax)
    sub_ax.set_title(title)
    sub_ax.invert_yaxis()

    short_labels = np.arange(1, len(ensemble_names) + 1)
    short_labels[-1] = 99

    sub_ax = ax[1]
    diff = model_col - target_col
    diff.index = short_labels

    colours = pd.Series('grey', index=diff.index)
    colours.loc[diff >= 2000] = 'skyblue'
    colours.loc[diff >= 5000] = 'dodgerblue'
    colours.loc[diff <= -2000] = 'lightsalmon'
    colours.loc[diff <= -5000] = 'tomato'

    diff.plot.barh(ax=sub_ax, color=colours)
    sub_ax.set_title("Error (Model - Target)")
    sub_ax.invert_yaxis()

    sub_ax = ax[2]
    perc_diff = diff / target_col
    perc_diff.index = short_labels

    colours = pd.Series('grey', index=perc_diff.index)
    colours.loc[perc_diff >= 0.1] = 'skyblue'
    colours.loc[perc_diff >= 0.25] = 'dodgerblue'
    colours.loc[perc_diff <= -0.1] = 'lightsalmon'
    colours.loc[perc_diff <= -0.25] = 'tomato'

    perc_diff.plot.barh(ax=sub_ax, color=colours)
    sub_ax.set_title(" % Error (Model - Target) / Target")
    sub_ax.invert_yaxis()

    plt.tight_layout()

    if fp is not None:
        fig.savefig(str(fp), dpi=dpi)

    return ax
