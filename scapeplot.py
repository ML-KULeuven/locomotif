import numpy as np
import matplotlib.pyplot as plt

def scape_plot(n, values, filename=None, figure=None):

    if figure is None:
        fig = plt.figure(figsize=(10, 10), frameon=True)
    else:
        fig = figure
    ax = fig.add_subplot()

    image = np.full((n, n), -np.inf)
    for (s, e, fitness) in values:
        length = e - s
        center = s + length // 2
        image[length, center] = fitness

    # Bottom left: (x0, y0), Bottom right: (x1, y1), Top:  (x2, y2)
    (x0, y0) = (0, 0)
    (x1, y1) = (n, 0)
    (x2, y2) = (n / 2.0, n)

    extent = (x0, x1, y0, y2)
    im = plt.imshow(image, cmap='viridis', origin='lower', aspect=1, interpolation='nearest', extent=extent)

    # Plot triangle sides
    ax.plot([x0, x1], [y0, y1], '-', linewidth=2, color='black')
    ax.plot([x0, x2], [y0, y2], '-', linewidth=2, color='black')
    ax.plot([x2, x1], [y2, y1], '-', linewidth=2, color='black')
    # plt.setp(ax.spines.values(), linewidth=2)

    # Set limits
    ax.set_xlim([x0, x1])
    ax.set_ylim([y0, y2])
    # Titles and labels
    ax.set_title("Fitness", fontsize=20)
    ax.set_xlabel("Segment center", fontsize=15)
    ax.set_ylabel("Segment length", fontsize=15)
    # Colorbar
    fig.colorbar(im, ax=ax)

    if filename:
        plt.savefig(filename)
        plt.close()
        fig, ax = None, None
    return fig, ax

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
