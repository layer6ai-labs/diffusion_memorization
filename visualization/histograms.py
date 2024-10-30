from typing import List, Optional
import functools

from cv2 import line
import matplotlib.pyplot as plt
import numpy as np
import PIL
import PIL.Image
import seaborn as sns
import pandas as pd

from .pretty import ColorTheme, FONT_FAMILY, hashlines, StyleDecorator, savable

def plot_bars_raw(
    x_values: List[str],
    group_bar_values: List[np.array],
    group_colors: List[str],
    group_labels: List[str],
    label_distance: float,
    x_label: Optional[str]=None,
    y_label: Optional[str]= None,
    figsize: tuple = (10, 6),
    bar_width: float = 0.35,
    fontsize: Optional[int] = None,
    xlim: Optional[tuple] = None,
    tick_fontsize: Optional[int] = None,
    no_legend: bool = False,
    legend_fontsize: Optional[int] = None,
    legend_loc: Optional[str] = None,
    clear_ylabel: bool = False,
    clear_yticks: bool = False,
    no_hashlines: bool = False,
):
    """
    Creates a barplot with the x-axis labelled according to x_values. 

    We also have multiple groups of bars, with the values for each group being stored in group_bar_values
    and the corresponding colors in group_colors. The labels for each group are stored in group_labels.
    """
     # Ensure all of the groups have the same number of values
    for group in group_bar_values:
        assert len(group) == len(x_values), "All groups must have the same number of values"
    assert len(group_bar_values) == len(group_labels) == len(group_colors), "Input group lists must have the same length"
    
    # create a dataframe
    df = pd.DataFrame(group_bar_values, index=group_labels).T
    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=figsize)

    # num_groups = len(group_bar_values)
    # num_bars_per_group = len(group_bar_values[0])
    x_indices = np.arange(len(x_values))  # Locations for the x-axis labels

    # Loop over each group and plot the bars with appropriate offsets
    for i, (bar_values, color, label) in enumerate(zip(group_bar_values, group_colors, group_labels)):
        bar_offset = i * bar_width  # Offset bars for each group
        ax.bar(
            x_indices + bar_offset,  # Shift the bar positions for each group
            bar_values,
            width=bar_width,
            color=color,
            label=label,
            hatch=hashlines[(i + 3) % len(hashlines)] if not no_hashlines else None,  # Apply hatch pattern
        )

    if y_label:
        y_label = y_label or 'count'
        ax.set_ylabel(f'{y_label}', fontsize=fontsize, fontdict={'family': FONT_FAMILY})
    
    # Remove default x-tick labels
    ax.set_xticks(x_indices + (len(group_bar_values) - 1) * bar_width / 2)
    ax.set_xticklabels([])  # Leave the x-ticks empty
    # Add shared labels under each group of bars
    for i, label in enumerate(x_values):
        ax.text(
            x_indices[i] + (len(group_bar_values) - 1) * bar_width / 2,  # Position in the middle of the grouped bars
            label_distance,  # Slightly below the x-axis
            label,  # The custom label for the bars
            ha='center', va='top', fontsize=tick_fontsize, fontdict={'family': FONT_FAMILY}
        )
        
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=fontsize, fontdict={'family': FONT_FAMILY})
    if xlim is not None:
        ax.set_xlim(xlim)
    
    if tick_fontsize is not None:
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
    
    if clear_yticks:
        ax.tick_params(axis='y', colors='white')
    if clear_ylabel:
        ax.yaxis.label.set_color('white')
        
    
    ax.legend(loc=legend_loc, prop={'family': FONT_FAMILY, 'size': legend_fontsize})
    if no_legend:
        ax.legend_.remove()

    return ax

def plot_kde_raw(
    x_values: List[np.array], 
    labels: List[str],
    colors: List[str],
    x_label: Optional[str]=None,
    y_label: Optional[str]= None,
    scale: int = 0,
    figsize: tuple = (10, 6),
    fontsize: Optional[int] = None,
    xlim: Optional[tuple] = None,
    skip_xticks: Optional[int] = None,
    tick_fontsize: Optional[int] = None,
    no_legend: bool = False,
    legend_fontsize: Optional[int] = None,
    legend_loc: Optional[str] = None,
    
    clear_ylabel: bool = False,
    clear_yticks: bool = False,
    show_scale: bool = True,
    no_hashlines: bool = False,
):
    """
    Plots KDE for given data.

    Parameters:
    - x_values: List of numpy arrays
    - labels: List of labels corresponding to x_values
    - colors: List of colors corresponding to x_values
    - scale: Integer value to scale the ylabel
    
    Returns:
    the ax for the possible decorators
    """
    
    # Ensure input lists are of the same length
    assert len(x_values) == len(labels) == len(colors), "Input lists must have the same length"
    
    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=figsize)

    idx = 0
    for x, label, color in zip(x_values, labels, colors):
        density = sns.kdeplot(x, bw_adjust=0.5, color=color).get_lines()[-1].get_data()
        ax.fill_between(density[0], 0, density[1] , color=color, label=label, alpha=0.5, hatch=hashlines[idx % len(hashlines)] if not no_hashlines else None)
        idx += 1
    
    y_label = y_label or 'Density'
    # Adjust y-axis label based on the scale
    # if scale != 0:
    ax.yaxis.set_major_formatter(lambda x, _: f'{x * 10 ** (scale):.1f}')
    if show_scale:
        y_label += f' $\\times 10^{{{scale}}}$'
    ax.set_ylabel(f'{y_label}', fontsize=fontsize, fontdict={'family': FONT_FAMILY})
    # else:
    #     ax.set_ylabel(f'{y_label}', fontsize=fontsize, fontdict={'family': FONT_FAMILY})
    
       
        
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=fontsize, fontdict={'family': FONT_FAMILY})
    if xlim is not None:
        ax.set_xlim(xlim)
    
    if tick_fontsize is not None:
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
    if skip_xticks:
        x_ticks = ax.get_xticks()
        if skip_xticks < 0:
            ax.set_xticks([])
        else:
            ax.set_xticks(x_ticks[::skip_xticks])  # Use every second tick
    
    if clear_yticks:
        ax.tick_params(axis='y', colors='white')
    if clear_ylabel:
        ax.yaxis.label.set_color('white')
        
    
    ax.legend(loc=legend_loc, prop={'family': FONT_FAMILY, 'size': legend_fontsize})
    if no_legend:
        ax.legend_.remove()
    
    return ax


@savable
@StyleDecorator(font_scale=1.5, style='ticks')
@functools.wraps(plot_kde_raw)
def plot_kde(*args, **kwargs):
    return plot_kde_raw(*args, **kwargs)
    
@savable
@StyleDecorator(font_scale=1.5, style='whitegrid', line_style='--')
@functools.wraps(plot_kde_raw)
def plot_kde_dotted(*args, **kwargs):
    return plot_kde_raw(*args, **kwargs)

@savable
@StyleDecorator(font_scale=1.5, style='ticks')
@functools.wraps(plot_bars_raw)
def plot_bars(*args, **kwargs):
    return plot_bars_raw(*args, **kwargs)
