import numpy as np
import random
import itertools
import scipy.stats as stats
import heapq
import matplotlib.pyplot as plt
import math
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import time
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple
from matplotlib.gridspec import GridSpec


def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    d_diags = [float(s.strip()) for s in lines[1: 101]]
    d_offs = [float(s.strip()) for s in lines[102: 202]]
    d_minors = [float(s.strip()) for s in lines[203: 303]]
    d_rows = [float(s.strip()) for s in lines[304: 404]]
    tau_diags = [float(s.strip()) for s in lines[405: 505]]
    tau_offs = [float(s.strip()) for s in lines[506: 606]]
    tau_minors = [float(s.strip()) for s in lines[607: 707]]
    tau_rows = [float(s.strip()) for s in lines[708: 808]]

    return d_diags, d_offs, d_minors, d_rows, tau_diags, tau_offs, tau_minors, tau_rows


if __name__ == '__main__':
    start_time = time.time()
    d_diags, d_offs, d_minors, d_rows, tau_diags, \
        tau_offs, tau_minors, tau_rows = read_file('barabasi_albert-d-u-homo-2026data')
    # d_offs, d_minors, d_rows, tau_diags, \
        # tau_offs, tau_minors, tau_rows = read_file('dolphins-s-u-homodata')
    # fig, (ax2, ax1, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.25)  # Ensure equal widths

    # Subplots
    ax2 = fig.add_subplot(gs[0, 0])  # Grid subplot
    ax1 = fig.add_subplot(gs[0, 1])  # Middle plot
    ax3 = fig.add_subplot(gs[0, 2])  # Right plot
    ax1.set_box_aspect(1)
    ax3.set_box_aspect(1)
    ax1.scatter(tau_diags, d_diags, edgecolors='r', facecolors='none', label='on-diagonal')
    ax1.scatter(tau_rows, d_rows, edgecolors='y', facecolors='none', label='row')
    ax1.scatter(tau_minors, d_minors, edgecolors='g', facecolors='none', label='minor')
    ax1.scatter(tau_offs, d_offs, edgecolors='b', facecolors='none', label='random')

    t_dmean, d_dmean, t_dstd, d_dstd = np.mean(tau_diags), np.mean(d_diags), np.std(tau_diags), np.std(d_diags)
    t_rmean, d_rmean, t_rstd, d_rstd = np.mean(tau_rows), np.mean(d_rows), np.std(tau_rows), np.std(d_rows)
    t_mmean, d_mmean, t_mstd, d_mstd = np.mean(tau_minors), np.mean(d_minors), np.std(tau_minors), np.std(d_minors)
    t_omean, d_omean, t_ostd, d_ostd = np.mean(tau_offs), np.mean(d_offs), np.std(tau_offs), np.std(d_offs)
    ax1.errorbar(t_dmean, d_dmean, xerr=t_dstd, yerr=d_dstd, color='r', ecolor='black',
                 elinewidth=1.5, capsize=3)
    ax1.errorbar(t_rmean, d_rmean, xerr=t_rstd, yerr=d_rstd, color='y', ecolor='black',
                 elinewidth=1.5, capsize=3)
    ax1.errorbar(t_mmean, d_mmean, xerr=t_mstd, yerr=d_mstd, color='g', ecolor='black',
                 elinewidth=1.5, capsize=3)
    ax1.errorbar(t_omean, d_omean, xerr=t_ostd, yerr=d_ostd, color='b', ecolor='black',
                 elinewidth=1.5, capsize=3)

    # plt.title('d vs τ for random minor, off-diagonal, on-diagonal, and row selections')
    ax1.set_ylabel('d', fontsize=10 * 1.5, fontstyle='italic')
    ax1.set_xlabel(r'$ \tau $', fontsize=10 * 1.5, fontstyle='italic')
    # ax1.legend(loc='upper left', fontsize=10 * 1.5)
    ax1.tick_params(axis='both', labelsize=15)
    # ax1.yticks(fontsize=10 * 1.5)
    # Create an 8x8 grid, initializing all cells as white
    ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
    ax1.set_xticklabels(['-0.2', '0', '0.2', '0.4', '0.6', '0.8'])
    ax1.set_ylim(bottom=-0.2)
    grid_size = 12
    grid = np.ones((grid_size, grid_size, 3))  # All cells start as white

    # Define colors
    red = [1, 0, 0]  # RGB for red
    green = [0, 1, 0]  # RGB for green
    blue = [0, 0, 1]  # RGB for blue
    yellow = [1, 1, 0]  # RGB for yellow

    # Set colors for specific positions based on the given coordinates
    grid[0, 0] = red
    grid[1, 1] = red
    grid[2, 2] = red
    grid[9, 9] = red
    grid[11, 11] = red

    grid[0, 1] = blue
    grid[1, 5] = blue
    grid[5, 9] = blue
    grid[9, 10] = blue
    grid[4, 11] = blue

    grid[2, 0] = yellow
    grid[2, 1] = yellow
    grid[2, 3] = yellow
    grid[2, 7] = yellow
    grid[2, 10] = yellow

    # Set green squares for (4<=i<=7, 4<=j<=7)
    for i in range(3, 8):
        for j in range(3, 8):
            grid[i, j] = green

    # Display the grid without ticks
    ax2.imshow(grid)

    # Draw the grid lines manually
    for i in range(0, grid_size + 1):
        ax2.axhline(i - 0.5, color='black', lw=2)  # Horizontal lines
        ax2.axvline(i - 0.5, color='black', lw=2)  # Vertical lines

    # Remove ticks (grid lines) from the axes
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax1.text(-0.06, 1, "(b)", transform=ax1.transAxes, fontsize=24, verticalalignment='top',
             horizontalalignment='right')
    ax2.text(-0.03, 1, "(a)", transform=ax2.transAxes, fontsize=24, verticalalignment='top',
             horizontalalignment='right')

    d_diags, d_offs, d_minors, d_rows, tau_diags, \
        tau_offs, tau_minors, tau_rows = read_file('dolphins-s-u-hetero-2026data')
    ax3.scatter(tau_diags, d_diags, edgecolors='r', facecolors='none', label='on-diagonal')
    ax3.scatter(tau_rows, d_rows, edgecolors='y', facecolors='none', label='row')
    ax3.scatter(tau_minors, d_minors, edgecolors='g', facecolors='none', label='minor')
    ax3.scatter(tau_offs, d_offs, edgecolors='b', facecolors='none', label='random')

    t_dmean, d_dmean, t_dstd, d_dstd = np.mean(tau_diags), np.mean(d_diags), np.std(tau_diags), np.std(d_diags)
    t_rmean, d_rmean, t_rstd, d_rstd = np.mean(tau_rows), np.mean(d_rows), np.std(tau_rows), np.std(d_rows)
    t_mmean, d_mmean, t_mstd, d_mstd = np.mean(tau_minors), np.mean(d_minors), np.std(tau_minors), np.std(d_minors)
    t_omean, d_omean, t_ostd, d_ostd = np.mean(tau_offs), np.mean(d_offs), np.std(tau_offs), np.std(d_offs)
    ax3.errorbar(t_dmean, d_dmean, xerr=t_dstd, yerr=d_dstd, color='r', ecolor='black',
                 elinewidth=1.5, capsize=3)
    ax3.errorbar(t_rmean, d_rmean, xerr=t_rstd, yerr=d_rstd, color='y', ecolor='black',
                 elinewidth=1.5, capsize=3)
    ax3.errorbar(t_mmean, d_mmean, xerr=t_mstd, yerr=d_mstd, color='g', ecolor='black',
                 elinewidth=1.5, capsize=3)
    ax3.errorbar(t_omean, d_omean, xerr=t_ostd, yerr=d_ostd, color='b', ecolor='black',
                 elinewidth=1.5, capsize=3)

    # plt.title('d vs τ for random minor, off-diagonal, on-diagonal, and row selections')
    ax3.set_ylabel('d', fontsize=10 * 1.5, fontstyle='italic')
    ax3.set_xlabel(r'$ \tau $', fontsize=10 * 1.5, fontstyle='italic')
    # ax3.legend(loc='upper left', fontsize=10 * 1.5)
    ax3.tick_params(axis='both', labelsize=15)
    ax3.text(-0.03, 1, "(c)", transform=ax3.transAxes, fontsize=24, verticalalignment='top',
            horizontalalignment='right')
    ax3.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax3.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8'])
    ax3.set_ylim(bottom=-0.2)

    # Create custom legend handles

    legend_handles = []
    colors = ["r", "g", "y", "b"]
    labels = ["Red", "Green", "Yellow", "Blue"]
    """
    legend_handles = []
    for color, label in zip(colors, labels):
        filled_square = mpatches.FancyBboxPatch((0, 0), 1, 1, color=color, boxstyle="square,pad=0.0")
        open_circle = mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                                    markersize=8, markerfacecolor='none', markeredgewidth=1)
        # Add a combined legend entry
        legend_handles.append(mlines.Line2D([], [], color=color, marker='s', linestyle='None',
                                            markersize=8, label=label))
        legend_handles.append(open_circle)

    # Add the unified legend on top
    fig.legend(handles=legend_handles, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1))
    """

    # Define colors and labels
    colors = ["r", "g", "y", "b"]
    labels = ["Diagonal", "Minor", "Row", "Random"]

    # Create legend handles
    legend_handles = []
    for color, label in zip(colors, labels):
        filled_square = mlines.Line2D([], [], color=color, marker='s', linestyle='None', markersize=10)
        open_circle = mlines.Line2D([], [], color=color, marker='o', markerfacecolor='none', linestyle='None',
                                    markersize=10)
        combined = (filled_square, open_circle, label)
        legend_handles.append(combined)

    fig.legend(
        handles=[(fs, oc) for fs, oc, lbl in legend_handles],
        labels=[lbl for fs, oc, lbl in legend_handles],
        handler_map={tuple: HandlerTuple(ndivide=None)},  # Treat (square, circle) as a single entry
        loc='upper center',
        ncol=4,
        bbox_to_anchor=(0.5, 0.95),
        columnspacing=1.5,  # Adjust spacing between columns
        handletextpad=0.8,  # Adjust spacing between handle and text
        fontsize=15,
        frameon=False
    )

    # plt.tight_layout()  # This ensures labels are visible without overlapping
    plt.subplots_adjust(wspace=0.2)
    plt.show()

    end_time = time.time()
    print(end_time - start_time)