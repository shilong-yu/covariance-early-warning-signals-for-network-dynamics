import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple


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
    d_uppers = [float(s.strip()) for s in lines[809: 909]]
    d_edges = [float(s.strip()) for s in lines[910: 1010]]
    tau_uppers = [float(s.strip()) for s in lines[1011: 1111]]
    tau_edges = [float(s.strip()) for s in lines[1112: 1212]]
    d_corr = [float(s.strip()) for s in lines[1213:1214]]
    tau_corr = [float(s.strip()) for s in lines[1215:1216]]

    return d_diags, d_offs, d_minors, d_rows, tau_diags, tau_offs, tau_minors, tau_rows, d_uppers, d_edges, tau_uppers, tau_edges, d_corr, tau_corr


if __name__ == '__main__':
    start_time = time.time()
    d_diags, d_offs, d_minors, d_rows, tau_diags, \
        tau_offs, tau_minors, tau_rows, d_uppers, d_edges, tau_uppers, tau_edges, d_corr, tau_corr = read_file('lizards-d-D-homo-down-2024')
    # d_offs, d_minors, d_rows, tau_diags, \
        # tau_offs, tau_minors, tau_rows = read_file('dolphins-s-u-homodata')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    ax3.set_box_aspect(1)
    ax1.scatter(tau_diags, d_diags, edgecolors='r', facecolors='none', label='on-diagonal')
    ax1.scatter(tau_rows, d_rows, edgecolors='y', facecolors='none', label='row')
    ax1.scatter(tau_minors, d_minors, edgecolors='g', facecolors='none', label='minor')
    ax1.scatter(tau_offs, d_offs, edgecolors='b', facecolors='none', label='random')
    ax1.scatter(tau_uppers, d_uppers, edgecolors='magenta', facecolors='none', label='upper')
    ax1.scatter(tau_edges, d_edges, edgecolors='brown', facecolors='none', label='edges')
    ax1.scatter(tau_corr, d_corr, marker='x', facecolors='k', label='corr', s=60)
    ax1.set_ylabel('d', fontsize=10 * 1.5, fontstyle='italic')
    ax1.set_xlabel('τ', fontsize=10 * 1.5, fontstyle='italic')
    ax1.set_xticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2])
    ax1.set_xticklabels(['-0.8', '-0.6', '-0.4', '-0.2', '0', '0.2'])
    ax1.set_ylim(bottom=-0.2)
    ax1.text(-0.03, 1, "(a)", transform=ax1.transAxes, fontsize=24, verticalalignment='top',
             horizontalalignment='right')
    ax2.text(-0.03, 1, "(b)", transform=ax2.transAxes, fontsize=24, verticalalignment='top',
             horizontalalignment='right')

    d_diags, d_offs, d_minors, d_rows, tau_diags, \
        tau_offs, tau_minors, tau_rows, d_uppers, d_edges, tau_uppers, tau_edges, d_corr, tau_corr = read_file('netsci-m-D-hetero-2024')
    ax2.scatter(tau_diags, d_diags, edgecolors='r', facecolors='none', label='on-diagonal')
    ax2.scatter(tau_rows, d_rows, edgecolors='y', facecolors='none', label='row')
    ax2.scatter(tau_minors, d_minors, edgecolors='g', facecolors='none', label='minor')
    ax2.scatter(tau_offs, d_offs, edgecolors='b', facecolors='none', label='random')
    ax2.scatter(tau_uppers, d_uppers, edgecolors='magenta', facecolors='none', label='upper')
    ax2.scatter(tau_edges, d_edges, edgecolors='brown', facecolors='none', label='edges')
    ax2.scatter(tau_corr, d_corr, marker='x', facecolors='k', label='corr', s=60)
    ax2.set_ylabel('d', fontsize=10 * 1.5, fontstyle='italic')
    ax2.set_xlabel('τ', fontsize=10 * 1.5, fontstyle='italic')
    ax2.set_xticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4])
    ax2.set_xticklabels(['-0.8', '-0.6', '-0.4', '-0.2', '0', '0.2', '0.4'] )
    ax2.set_ylim(bottom=-0.2)
    
    d_diags, d_offs, d_minors, d_rows, tau_diags, \
        tau_offs, tau_minors, tau_rows, d_uppers, d_edges, tau_uppers, tau_edges, d_corr, tau_corr = read_file('surfers-d-D-hetero-2024')
    # d_offs, d_minors, d_rows, tau_diags, \
        # tau_offs, tau_minors, tau_rows = read_file('dolphins-s-u-homodata')
    ax3.scatter(tau_diags, d_diags, edgecolors='r', facecolors='none', label='on-diagonal')
    ax3.scatter(tau_rows, d_rows, edgecolors='y', facecolors='none', label='row')
    ax3.scatter(tau_minors, d_minors, edgecolors='g', facecolors='none', label='minor')
    ax3.scatter(tau_offs, d_offs, edgecolors='b', facecolors='none', label='random')
    ax3.scatter(tau_uppers, d_uppers, edgecolors='magenta', facecolors='none', label='upper')
    ax3.scatter(tau_edges, d_edges, edgecolors='brown', facecolors='none', label='edges')
    ax3.scatter(tau_corr, d_corr, marker='x', facecolors='k', label='corr', s=60)
    ax3.set_ylabel('d', fontsize=10 * 1.5, fontstyle='italic')
    ax3.set_xlabel('τ', fontsize=10 * 1.5, fontstyle='italic')

    ax3.text(-0.05, 1, "(c)", transform=ax3.transAxes, fontsize=24, verticalalignment='top',
            horizontalalignment='right')
    ax3.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8])
    ax3.set_xticklabels(['-0.2', '0', '0.2', '0.4', '0.6', '0.8'])
    ax3.set_ylim(bottom=-0.2)

    # Define colors and labels
    colors = ["r", "g", "y", "b", "magenta", "brown", "k"]
    labels = ["Diagonal", "Minor", "Row", "Random", "Upper triangular", "Edge", "Correlation rank"]

    # Create legend handles
    legend_handles = []
    for color, label in zip(colors, labels):
        if label == "Correlation rank":  # Example for a single square marker legend
            filled_x = mlines.Line2D([], [], color=color, marker='x', linestyle='None', markersize=10)
            legend_handles.append((filled_x, label))  # Append a tuple with just the marker and label
        else:
            # filled_square = mlines.Line2D([], [], color=color, marker='s', linestyle='None', markersize=10)
            open_circle = mlines.Line2D([], [], color=color, marker='o', markerfacecolor='none', linestyle='None',
                                        markersize=10)
            legend_handles.append((open_circle, label))  # Append a tuple with both markers and label

    # Separate the legend handles into two rows
    first_row_handles = legend_handles[:4]  # First three entries
    second_row_handles = legend_handles[4:]  # Remaining entries

    # Combine handles and labels for the two rows
    handles = ([fs if isinstance(fs, tuple) else (fs,) for fs, lbl in first_row_handles] +
               [fs if isinstance(fs, tuple) else (fs,) for fs, lbl in second_row_handles])
    labels = ([lbl for fs, lbl in first_row_handles] +
              [lbl for fs, lbl in second_row_handles])

    # fig, ax5 = plt.subplots()

    # Add the first row legend
    first_row_handles_only = [fs if isinstance(fs, tuple) else (fs,) for fs, lbl in first_row_handles]
    first_row_labels = [lbl for fs, lbl in first_row_handles]
    legend1 = fig.legend(
        handles=first_row_handles_only,
        labels=first_row_labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},  # Treat (square, circle) as a single entry
        loc='upper center',
        bbox_to_anchor=(0.5, 1),  # Adjust position for the first row
        ncol=4,
        columnspacing=1.5,
        handletextpad=0.1,
        fontsize=15,
        frameon=False
    )

    # Add the second row legend
    second_row_handles_only = [fs if isinstance(fs, tuple) else (fs,) for fs, lbl in second_row_handles]
    second_row_labels = [lbl for fs, lbl in second_row_handles]
    legend2 = fig.legend(
        handles=second_row_handles_only,
        labels=second_row_labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},  # Treat (square, circle) as a single entry
        loc='upper center',
        bbox_to_anchor=(0.5, 0.95),  # Adjust position for the second row
        ncol=3,
        columnspacing=1.5,
        handletextpad=0.1,
        fontsize=15,
        frameon=False
    )

    # Add legends to the figure
    fig.add_artist(legend1)
    fig.add_artist(legend2)

    # plt.tight_layout()  # This ensures labels are visible without overlapping
    plt.subplots_adjust(wspace=0.25)
    plt.show()

    end_time = time.time()
    print(end_time - start_time)
