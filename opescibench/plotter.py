import numpy as np
from math import log, floor, ceil
from os import path, makedirs
from collections import Mapping
import matplotlib as mpl
# The below is needed on certain clusters
# mpl.use("Agg")
import matplotlib.pyplot as plt


__all__ = ['Plotter']


def scale_limits(minval, maxval, base, type='log'):
    """ Compute axis values from min and max values """
    if type == 'log':
        basemin = floor(log(minval, base))
        basemax = ceil(log(maxval, base))
    else:
        basemin = floor(float(minval) / base)
        basemax = ceil(float(maxval) / base)
    nvals = basemax - basemin + 1
    dtype = np.int32 if abs(minval) > 1. else np.float32
    basevals = np.linspace(basemin, basemax, nvals, dtype=dtype)
    if type == 'log':
        return dtype(base) ** basevals
    else:
        return dtype(base) * basevals


class Plotter(object):
    """ Plotting utility that provides data and basic diagram utilities. """

    figsize = (6, 4)
    dpi = 300
    marker = ['D', 'o', '^', 'v']

    def __init__(self, plotdir='plots'):
        self.plotdir = plotdir

    def create_figure(self, figname):
        fig = plt.figure(figname, figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111)
        return fig, ax

    def set_xaxis(self, axis, label, values=None):
        if values is not None:
            axis.set_xlim(values[0], values[-1])
            axis.set_xticks(np.linspace(values[0], values[-1], len(values)))
            axis.set_xticklabels(values)
        axis.set_xlabel(label)

    def set_yaxis(self, axis, label, values=None):
        if values is not None:
            axis.set_ylim(values[0], values[-1])
            axis.set_yticks(np.linspace(values[0], values[-1], len(values)))
            axis.set_yticklabels(values)
        axis.set_ylabel(label)

    def save_figure(self, figure, figname):
        if not path.exists(self.plotdir):
            makedirs(self.plotdir)
        figpath = path.join(self.plotdir, figname)
        print "Plotting %s " % figpath
        figure.savefig(figpath, format='pdf', facecolor='white',
                       orientation='landscape', bbox_inches='tight')

    def plot_strong_scaling(self, figname, nprocs, time, save=True,
                            xlabel='Number of processors', ylabel='Wall time (s)'):
        """ Plot a strong scaling diagram and according parallel efficiency.

        :param nprocs: List of processor counts or a dict mapping labels to processors
        :param time: List of timings or a dict mapping labels to timings
        """
        assert(isinstance(nprocs, Mapping) == isinstance(time, Mapping))
        fig, ax = self.create_figure(figname)

        if isinstance(nprocs, Mapping):
            for i, label in enumerate(nprocs.keys()):
                ax.loglog(nprocs[label], time[label], label=label,
                          linewidth=2, linestyle='solid', marker=self.marker[i])
            ymin = min([min(t) for t in time.values() if len(t) > 0])
            ymax = max([max(t) for t in time.values() if len(t) > 0])
        else:
            ax.loglog(nprocs, time, linewidth=2, linestyle='solid', marker=self.marker[0])
            ymin = min(time)
            ymax = max(time)

        # Add legend if labels were used
        if isinstance(nprocs, Mapping):
            ax.legend(loc='best', ncol=4, fancybox=True, fontsize=10)
        self.set_xaxis(ax, xlabel, values=nprocs)
        self.set_yaxis(ax, ylabel, values=scale_limits(ymin, ymax, type='log', base=2.))
        return self.save_figure(fig, figname) if save else fig, ax

    def plot_efficiency(self, figname, nprocs, time, save=True,
                        xlabel='Number of processors', ylabel='Parallel efficiency'):
        assert(isinstance(nprocs, Mapping) == isinstance(time, Mapping))
        fig, ax = self.create_figure(figname)

        if isinstance(nprocs, Mapping):
            for i, label in enumerate(nprocs.keys()):
                ax.loglog(nprocs[label], time[label] / time[label][0], label=label,
                          linewidth=2, linestyle='solid', marker=self.marker[i])
        else:
            ax.semilogx(nprocs, (time[0] /time) / nprocs, linewidth=2,
                        linestyle='solid', marker=self.marker[0])

        # Add legend if labels were used
        if isinstance(nprocs, Mapping):
            ax.legend(loc='best', ncol=4, fancybox=True, fontsize=10)
        self.set_xaxis(ax, xlabel, values=nprocs)
        yvals = np.linspace(0., 1.2, 7)
        self.set_xaxis(ax, xlabel, values=nprocs)
        self.set_yaxis(ax, ylabel, values=yvals)
        return self.save_figure(fig, figname) if save else fig, ax

    def plot_error_cost(self, figname, error, time, annotations=None, save=True,
                        xlabel='Error in L2 norm', ylabel='Wall time (s)'):
        """ Plot an error cost diagram for the given error and time data.

        :param figname: Name of output file
        :param error: List of error values or a dict mapping labels to values lists
        :param time: List of time measurements or a dict mapping labels to values lists
        :param annotations: Optional list of point annotations
        :param save: Whether to save the plot; if False a tuple (fig, axis) is returned
        """
        assert(isinstance(error, Mapping) == isinstance(time, Mapping))
        fig, ax = self.create_figure(figname)

        if isinstance(error, Mapping):
            for i, label in enumerate(error.keys()):
                ax.loglog(error[label], time[label], label=label,
                          linewidth=2, linestyle='solid', marker=self.marker[i])
            if annotations is not None:
                for label in annotations:
                    for x, y, a in zip(error[label], time[label], annotations[label]):
                        plt.annotate(a, xy=(x, y), xytext=(4, 2),
                                     textcoords='offset points', size=8)
            ymin = min([min(t) for t in time.values()])
            ymax = max([max(t) for t in time.values()])
        else:
            ax.loglog(error, time, linewidth=2, linestyle='solid', marker=self.marker[0])
            ymin = min(time)
            ymax = max(time)

        if isinstance(error, Mapping):
            ax.legend(loc='best', ncol=4, fancybox=True, fontsize=10)
        self.set_xaxis(ax, xlabel)
        self.set_yaxis(ax, ylabel, values=scale_limits(ymin, ymax, type='log', base=2.))
        return self.save_figure(fig, figname) if save else fig, ax

    def plot_roofline(self, problem, flops, intensity,
                      max_bw=None, max_flops=None, save=True,
                      xlabel='Operational intensity (Flops/Byte)',
                      ylabel='Performance (GFlops/s)'):
        """ Plot performance on a roofline graph with given limits.

        :param problem: Dict describing the problem instance
        :param flops: Dict of labels to flops performed in MFlops
        :param intensity: Dict of labels to operational intensity values (Flops/B)
        :param max_bw: Maximum achievable memory bandwidth; determines roofline slope
        :param max_flops: Maximum achievable flop rate; determines roof of the diagram
        :param save: Whether to save the plot; if False a tuple (fig, axis) is returned
        """
        assert(isinstance(flops, Mapping) and isinstance(intensity, Mapping))
        assert(max_bw is not None and max_flops is not None)

        name = problem["name"]
        compiler = problem["compiler"]
        grid = problem["grid"]
        spacing = problem["spacing"]
        space_order = str(problem["space_order"])
        time_order = str(problem["time_order"])

        figname = "%s_dim%s_so%s_to%s.pdf" % (name, grid, space_order, time_order)
        fig, ax = self.create_figure(figname)

        ax.set_title("%s %s so=%s to=%s" % (name, grid, space_order, time_order))

        # Derive axis values for flops rate and operational intensity
        xvals = scale_limits(min(intensity.values()),
                             max(intensity.values()),
                             base=2., type='log')
        yvals = scale_limits(min(flops.values()),
                             max(flops.values()),
                             base=10., type='log')
        roofline = xvals * max_bw
        roofline[roofline > max_flops] = max_flops
        idx = (roofline >= max_flops).argmax()
        x_roofl = np.insert(xvals, idx, max_flops / max_bw)
        roofline = np.insert(roofline, idx, max_flops)
        ax.loglog(x_roofl, roofline, 'k-')

        # Insert roofline points
        for label in flops.keys():
            oi = intensity[label]
            ax.loglog(oi, flops[label], 'k%s' % self.marker[0])
            ax.plot([oi, oi], [yvals[0], min(oi * max_bw, max_flops)], 'k:')
            plt.annotate(label, xy=(oi, flops[label]), xytext=(2, -13),
                         rotation=-45, textcoords='offset points', size=10)
        self.set_xaxis(ax, xlabel, values=xvals)
        self.set_yaxis(ax, ylabel, values=yvals)
        # Convert MFlops to GFlops in plot
        ax.set_yticklabels(yvals / 1000)
        return self.save_figure(fig, figname) if save else fig, ax


    def plot_comparison(self, figname, mode, time, save=True,
                            xlabel='Execution mode', ylabel='Wall time (s)'):
        """Plot bar chart comparison between different execution modes.

        :param mode: List of modes or a dict mapping labels to processors
        :param time: List of timings or a dict mapping labels to timings
        """
        assert(isinstance(mode, Mapping) == isinstance(time, Mapping))
        fig, ax = self.create_figure(figname)
        offsets = np.arange(len(mode))
        width = 0.8

        if isinstance(mode, Mapping):
            raise NotImplementedError('Custom labels not yet supported for bar chart comparison')
        else:
            ax.bar(offsets + .1, time, width)
            ymin = min(time)
            ymax = max(time)

        # Add legend if labels were used
        if isinstance(mode, Mapping):
            ax.legend(loc='best', ncol=4, fancybox=True, fontsize=10)
        ax.set_xticks(offsets + .5)
        ax.set_xticklabels(mode)
        self.set_yaxis(ax, ylabel, values=scale_limits(ymin, ymax, type='log', base=2.))
        return self.save_figure(fig, figname) if save else fig, ax
