import matplotlib.pyplot as plt
import numpy as np
from math import log, floor, ceil
from os import path
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
        self.parser.add_argument('--plottype', choices=('error', 'comparison'), default='error',
                                 help='Type of plot to generate: error-cost or barchart comparison')
        self.parser.add_argument('-i', '--resultsdir', default='results',
                                 help='Directory containing results')
        self.parser.add_argument('-o', '--plotdir', default='plots',
                                 help='Directory to store generated plots')

    def plot_error_cost(self, figname, error, time, save=True):
        """ Plot an error cost diagram for the given error and time data.

        :param figname: Name of output file
        :param error: List of error values or a dict mapping labels to values lists
        :param time: List of time measurements or a dict mapping labels to values lists
        :param save: Whether to save the plot; if False a tuple (fig, axis) is returned
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
            figpath = path.join(self.bench.args.plotdir, figname)
            print "Plotting error-cost plot: %s " % figpath
            fig.savefig(figpath, format='pdf', facecolor='white',
                        orientation='landscape', bbox_inches='tight')
        else:
            return fig, ax
