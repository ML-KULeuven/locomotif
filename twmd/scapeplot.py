import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    'text.latex.preamble': r"\usepackage{amsmath}"
})

def scape_plot(n, values, filename=None, colorbar=False, vmin=0, vmax=1):
    cax = None
    figsize = (5, 5)
    rect = [0.2, 0.2, 0.7, 0.7]
    if colorbar:
        fig, ax, cax = split_figure_vertically(figsize, 1, rect, [0., 0.2, 0.2, 0.7])
    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect)
    
    image = np.full((n, n), -np.inf)
    for (s, e, value) in values:
        s, e = int(s), int(e)
        length = e - s
        center = s + length // 2
        image[length-1, center] = value

    # Bottom left: (x0, y0), Bottom right: (x1, y1), Top:  (x2, y2)
    (x0, y0) = (0      , 1)
    (x1, y1) = (n - 1  , 1)
    (x2, y2) = (n / 2.0, n)

    extent = (x0-0.5, x1+0.5, y0-0.5, y2+0.5)
    im = ax.imshow(image, cmap='inferno', origin='lower', aspect=1, interpolation='nearest', extent=extent, vmin=vmin, vmax=vmax)

    # Plot triangle sides
    ax.plot([x0-0.5, x1+0.5], [y0-0.5, y1-0.5], '-', linewidth=1, color='black')
    ax.plot([x0-0.5, x2], [y0-0.5, y2+0.5], '-', linewidth=1, color='black')
    ax.plot([x2, x1+0.5], [y2+0.5, y1-0.5], '-', linewidth=1, color='black')
    # plt.setp(ax.spines.values(), linewidth=2)

    spines = ['left', 'bottom']
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 15))  # outward by 15 points
        else:
            spine.set_color('none')

    # Colorbar
    if colorbar:
        cbar = fig.colorbar(im, cax=cax)
        cax.tick_params(labelsize=13)
    
    # Set limits
    ax.set_xlim([x0-0.5-0.005*n, x1+0.5+0.005*n])
    ax.set_ylim([y0-0.5-0.005*n, y2+0.5+0.005*n])
    # Titles and labels
    # ax.set_title("Fitness", fontsize=16)
    ax.set_xlabel(r"$center(\alpha)$", fontsize=13)
    ax.set_ylabel(r"$|\alpha|$", fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=13)

    
    if filename:
        plt.savefig(filename)
        plt.close()
        fig, ax, cax = None, None, None
    return fig, ax, cax

def plot_segment(ax, s, e, color='black'):
    length = e - s
    center = s + length // 2
    ax.scatter(center + 0.5, length + 0.5, s=20, c=color)
    return ax

def plot_segments(ax, segments):
    for (s, e) in segments:
        plot_segment(ax, s, e)
    return ax

def plot_subsegments(ax, s, e):
    length = e - s
    center = s + length // 2

    (x0, y0) = (s, 0)
    (x1, y1) = (center, length)
    (x2, y2) = (e, 0)

    ax.plot([x0, x1], [y0, y1], ':', linewidth=1, color='black')
    ax.plot([x1, x2], [y1, y2], ':', linewidth=1, color='black')
    return ax

def plot_disjoint_segments(ax, s, e, n):
    (x0, y0) = (s / 2.0, s)
    (x1, y1) = (s, 0)
    ax.plot([x0, x1], [y0, y1], ':', linewidth=1, color='black')

    (x0, y0) = (e, 0)
    (x1, y1) = ((n + e) / 2.0, n - e)
    ax.plot([x0, x1], [y0, y1], ':', linewidth=1, color='black')
    return ax


def split_figure_vertically(figsize_1, additional_width, rect_1, rect_2):
    old_width_1 = figsize_1[0]
    new_width = old_width_1 + additional_width
    factor_1 = old_width_1 / new_width
    factor_2 = additional_width / new_width
    
    figsize = (new_width, figsize_1[1])
    
    fig = plt.figure(figsize=figsize)
    
    rect_1[0] *= factor_1
    rect_1[2] *= factor_1
    
    rect_2[0] *= factor_2
    rect_2[2] *= factor_2
    rect_2[0] += factor_1
    
    ax1 = fig.add_axes(rect_1)
    ax2 = fig.add_axes(rect_2)
    return fig, ax1, ax2