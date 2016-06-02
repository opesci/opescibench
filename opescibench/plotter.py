import matplotlib.pyplot as plt
import numpy as np
from math import log, floor, ceil
from os import path, makedirs
from collections import Mapping

__all__ = ['Plotter']


class Plotter(object):
    """ Plotting utility that provides data and basic diagram utilities. """

    figsize = (6, 4)
    dpi = 300
    marker = ['D', 'o', '^', 'v']

    def __init__(self, benchmark, parser):
        self.bench = benchmark

        self.parser = parser
        self.parser.add_argument('--plottype', choices=('error', 'roofline'), default='error',
                                 help='Type of plot to generate: error-cost or roofline')
        self.parser.add_argument('-i', '--resultsdir', default='results',
                                 help='Directory containing results')
        self.parser.add_argument('-o', '--plotdir', default='plots',
                                 help='Directory to store generated plots')
        self.parser.add_argument('--max-bw', metavar='max_bw', type=float,
                                 help='Maximum memory bandwidth for roofline plots')
        self.parser.add_argument('--max-flops', metavar='max_flops', type=float,
                                 help='Maximum flop rate for roofline plots')

    def save_figure(self, figure, figname):
        plotdir = self.bench.args.plotdir
        if not path.exists(plotdir):
            makedirs(plotdir)
        figpath = path.join(plotdir, figname)
        print "Plotting %s " % figpath
        figure.savefig(figpath, format='pdf', facecolor='white',
                       orientation='landscape', bbox_inches='tight')

    def plot_error_cost(self, figname, error, time, annotations=None, save=True):
        """ Plot an error cost diagram for the given error and time data.

        :param figname: Name of output file
        :param error: List of error values or a dict mapping labels to values lists
        :param time: List of time measurements or a dict mapping labels to values lists
        :param save: Whether to save the plot; if False a tuple (fig, axis) is returned
        :param annotations: Optional list of point annotations
        """
        assert(isinstance(error, Mapping) == isinstance(time, Mapping))
        fig = plt.figure(figname, figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111)

        if isinstance(error, Mapping):
            for i, label in enumerate(error.keys()):
                ax.loglog(error[label], time[label], label=label,
                          linewidth=2, linestyle='solid', marker=self.marker[i])
            ymin = floor(log(min([min(t) for t in time.values() if len(t) > 0]), 2.))
            ymax = ceil(log(max([max(t) for t in time.values() if len(t) > 0]), 2.))
            if annotations is not None:
                for label in annotations:
                    for x, y, a in zip(error[label], time[label], annotations[label]):
                        plt.annotate(a, xy=(x, y), xytext=(4, 2),
                                     textcoords='offset points', size=8)
        else:
            ax.loglog(error, time, linewidth=2, linestyle='solid', marker=self.marker[0])
            ymin = floor(log(min(time), 2.))
            ymax = ceil(log(max(time), 2.))

        # Add legend if labels were used
        if isinstance(error, Mapping):
            ax.legend(loc='best', ncol=4, fancybox=True, fontsize=10)
        # Enforce power-of-2 log scale for time
        yvals = 2.0 ** np.linspace(ymin, ymax, ymax-ymin+1)
        ax.set_ylim(yvals[0], yvals[-1])
        ax.set_yticks(yvals)
        ax.set_yticklabels(yvals)
        ax.set_ylabel('Wall time (s)')
        ax.set_xlabel('Error in L2 norm')
        if save:
            self.save_figure(fig, figname)
        else:
            return fig, ax

    def plot_roofline(self, figname, flopss, intensity, max_bw=None, max_flops=None, save=True):
        """ Plot performance on a roofline graph with given limits.

        :param figname: Name of output file
        :param flopss: Dict of labels to flop rates (MFlops/s)
        :param intensity: Dict of labels to operational intensity values (Flops/B)
        :param max_bw: Maximum achievable memory bandwidth; determines roofline slope
        :param max_flops: Maximum achievable flop rate; determines roof of the diagram
        :param save: Whether to save the plot; if False a tuple (fig, axis) is returned
        """
        assert(isinstance(flopss, Mapping) and isinstance(intensity, Mapping))
        fig = plt.figure(figname, figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111)

        max_bw = max_bw or self.bench.args.max_bw
        max_flops = max_flops or self.bench.args.max_flops
        assert(max_bw is not None and max_flops is not None)

        # Derive axis values for flops rate and operational intensity
        xmin = floor(log(min(intensity.values()), 2.))
        xmax = ceil(log(max(intensity.values()), 2.))
        xvals = 2 ** np.linspace(xmin, xmax, xmax - xmin + 1)
        ymin = floor(log(min(flopss.values()), 10.))
        ymay = ceil(log(max(flopss.values()), 10.))
        yvals = 10. ** np.linspace(ymin, ymay, ymay - ymin + 1)
        # Derive roofline and insert the explicit crossover point
        roofline = xvals * max_bw
        roofline[roofline > max_flops] = max_flops
        idx = (roofline >= max_flops).argmax()
        x_roofl = np.insert(xvals, idx, max_flops / max_bw)
        roofline = np.insert(roofline, idx, max_flops)
        ax.loglog(x_roofl, roofline, 'k-')

        # Insert roofline points
        for label in flopss.keys():
            oi = intensity[label]
            ax.loglog(oi, flopss[label], 'k%s' % self.marker[0])
            ax.plot([oi, oi], [ymin, min(oi * max_bw, max_flops)], 'k:')
            plt.annotate(label, xy=(oi, flopss[label]), xytext=(3, -20),
                         rotation=-90, textcoords='offset points', size=10)

        # Enforce axis limits, set values and labels
        ax.set_ylim(yvals[0], yvals[-1])
        ax.set_yticks(yvals)
        ax.set_yticklabels(yvals / 1000)
        ax.set_ylabel('Performance (GFlops/s)')
        ax.set_xlim(xvals[0], xvals[-1])
        ax.set_xticks(xvals)
        ax.set_xticklabels(xvals)
        ax.set_xlabel('Operational intensity (Flops/Byte)')
        if save:
            self.save_figure(fig, figname)
        else:
            return fig, ax
