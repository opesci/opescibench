import numpy as np
from math import log, floor, ceil
from os import path, makedirs
from collections import Mapping, namedtuple, defaultdict, OrderedDict
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # The below is needed on certain clusters
    # mpl.use("Agg")
    from matplotlib.ticker import FormatStrFormatter

    # Adjust font size
    font = {'size'   : 10}
    mpl.rc('font', **font)
except:
    mpl = None
    plt = None
try:
    import brewer2mpl as b2m
except ImportError:
    b2m = None
from opescibench.utils import bench_print


__all__ = ['Plotter', 'LinePlotter', 'RooflinePlotter', 'BarchartPlotter']


def scale_limits(minval, maxval, base, type='log'):
    """ Compute axis values from min and max values """
    if type == 'log':
        basemin = floor(log(minval, base))
        basemax = ceil(log(maxval, base))
    else:
        basemin = floor(float(minval) / base)
        basemax = ceil(float(maxval) / base)
    nvals = basemax - basemin + 1
    dtype = np.float32
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

    if b2m is not None:
        color = b2m.get_map('Set2', 'qualitative', 6).hex_colors
    else:
        color = ['r', 'b', 'g', 'y']

    fonts = {'title': 7, 'axis': 8, 'minorticks': 3, 'legend': 7}

    def __init__(self, plotdir='plots'):
        if mpl is None or plt is None:
            bench_print("Matplotlib/PyPlot not found - unable to plot.")
            raise ImportError("Could not import matplotlib or pyplot")
        self.plotdir = plotdir

    def create_figure(self, figname):
        fig = plt.figure(figname, figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111)
        return fig, ax

    def set_xaxis(self, axis, label, values=None, dtype=np.float32):
        if values is not None:
            values = np.array(values).astype(dtype)
            axis.set_xlim(values[0], values[-1])
            axis.set_xticks(values)
            axis.set_xticklabels(values, fontsize=self.fonts['axis'])
        axis.set_xlabel(label, fontsize=self.fonts['axis'])

    def set_yaxis(self, axis, label, values=None, dtype=np.float32):
        if values is not None:
            values = np.array(values).astype(dtype)
            axis.set_ylim(values[0], values[-1])
            axis.set_yticks(values)
            axis.set_yticklabels(values, fontsize=self.fonts['axis'])
        axis.set_ylabel(label, fontsize=self.fonts['axis'])

    def save_figure(self, figure, figname):
        if not path.exists(self.plotdir):
            makedirs(self.plotdir)
        figpath = path.join(self.plotdir, figname)
        bench_print("Plotting %s " % figpath)
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
            ax.legend(loc='best', ncol=4, fancybox=True, fontsize=self.fonts['legend'])
        self.set_xaxis(ax, xlabel, values=nprocs, dtype=np.int32)
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
            ax.legend(loc='best', ncol=4, fancybox=True, fontsize=self.fonts['legend'])
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
            ax.legend(loc='best', ncol=4, fancybox=True, fontsize=self.fonts['legend'])
        self.set_xaxis(ax, xlabel)
        self.set_yaxis(ax, ylabel, values=scale_limits(ymin, ymax, type='log', base=2.))
        return self.save_figure(fig, figname) if save else fig, ax

    def plot_roofline(self, figname, flopss, intensity, max_bw=None, max_flops=None,
                      save=True, xlabel='Operational intensity (Flops/Byte)',
                      ylabel='Performance (GFlops/s)', title=None):
        """ Plot performance on a roofline graph with given limits.

        :param figname: Name of output file
        :param flopss: Dict of labels to flop rates (GFlops/s)
        :param intensity: Dict of labels to operational intensity values (Flops/B)
        :param max_bw: Maximum achievable memory bandwidth (GB/s); determines roofline slope
        :param max_flops: Maximum achievable flop rate (GFlops/s); determines roof of the diagram
        :param save: Whether to save the plot; if False a tuple (fig, axis) is returned
        """
        assert(isinstance(flopss, Mapping) and isinstance(intensity, Mapping))
        fig, ax = self.create_figure(figname)
        if title is not None:
            ax.set_title(title)

        assert(max_bw is not None and max_flops is not None)

        # Derive axis values for flops rate and operational intensity
        xmax = max(intensity.values() + [float(max_flops) / max_bw])
        ymax = max(flopss.values() + [max_flops])
        xvals = scale_limits(min(intensity.values()), xmax, base=2., type='log')
        yvals = scale_limits(min(flopss.values()), ymax, base=2., type='log')

        # Create the roofline
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
            ax.plot([oi, oi], [yvals[0], min(oi * max_bw, max_flops)], 'k:')
            plt.annotate(label, xy=(oi, flopss[label]), xytext=(2, -13),
                         rotation=-45, textcoords='offset points', size=8)
        self.set_xaxis(ax, xlabel, values=xvals, dtype=np.int32)
        self.set_yaxis(ax, ylabel, values=yvals, dtype=np.int32)
        # Convert MFlops to GFlops in plot
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
            ax.legend(loc='best', ncol=4, fancybox=True, fontsize=self.fonts['legend'])
        ax.set_xticks(offsets + .5)
        ax.set_xticklabels(mode)
        self.set_yaxis(ax, ylabel, values=scale_limits(ymin, ymax, type='log', base=2.))
        return self.save_figure(fig, figname) if save else fig, ax


class LinePlotter(Plotter):
    """Line plotter for generating scaling or error-cost plots

    :params figname: Name of output file
    :params plotdir: Directory to store the plot in
    :params title: Plot title to be printed on top

    Example usage:

    with LinePlotter(figname=..., plotdir=...) as plot:
        plot.add_line(y_values, x_values, label='Strong scaling')
    """

    def __init__(self, figname='plot', plotdir='plots', title=None,
                 plot_type='loglog', xlabel=None, ylabel=None,
                 xvalues=None, yvalues=None,
                 xtype=np.int32, ytype=np.int32, xbase=2., ybase=2.):
        super(LinePlotter, self).__init__(plotdir=plotdir)
        self.figname = figname
        self.title = title
        self.legend = OrderedDict()
        self.plot_type = plot_type
        self.xlabel = xlabel or 'Number of processors'
        self.ylabel = ylabel or 'Wall time (s)'
        self.xlim = None
        self.ylim = None
        self.xvalues = xvalues
        self.yvalues = yvalues
        self.xtype = xtype
        self.ytype = ytype
        self.xbase = xbase
        self.ybase = ybase

    def __enter__(self):
        self.fig, self.ax = self.create_figure(self.figname)
        self.plot = getattr(self.ax, self.plot_type)
        if self.title is not None:
            self.ax.set_title(title)
        return self

    def __exit__(self, *args):
        # Set axis labelling and generate plot file
        if self.xlim:
            if self.xvalues is None:
                plttype = 'log' if self.plot_type in ['loglog', 'semilogx'] else 'lines'
                xvals = scale_limits(self.xlim[0], self.xlim[1], base=self.xbase, type=plttype)
            else:
                xvals = self.xvalues
            self.set_xaxis(self.ax, self.xlabel, values=xvals, dtype=self.xtype)
        if self.ylim:
            if self.yvalues is None:
                plttype = 'log' if self.plot_type in ['loglog', 'semilogy'] else 'lines'
                yvals = scale_limits(self.ylim[0], self.ylim[1], base=self.ybase, type=plttype)
            else:
                yvals = self.yvalues
            self.set_yaxis(self.ax, self.ylabel, values=yvals, dtype=self.ytype)
        # Add legend if labels were used
        if len(self.legend) > 0:
            self.ax.legend(self.legend, loc='best', ncol=2,
                           fancybox=True, fontsize=10)
        self.save_figure(self.fig, self.figname)

    def add_line(self, xvalues, yvalues, label=None, style=None):
        """Adds a single line to the plot of from a set of measurements

        :param yvalue: List of Y values of the  measurements
        :param xvalue: List of X values of the  measurements
        :param label: Optional legend label for data line
        :param style: Plotting style to use, defaults to black line ('-k')
        """
        style = style or 'k-'
        # Update mai/max values for axis limits
        xv_lim = (min(xvalues), max(xvalues))
        self.xlim = (min(xv_lim[0], self.xlim[0]) if self.xlim else xv_lim[0],
                     max(xv_lim[1], self.xlim[1]) if self.xlim else xv_lim[1])
        yv_lim = (min(yvalues), max(yvalues))
        self.ylim = (min(yv_lim[0], self.ylim[0]) if self.ylim else yv_lim[0],
                     max(yv_lim[1], self.ylim[1]) if self.ylim else yv_lim[1])
        self.plot(xvalues, yvalues, style, label=label, linewidth=2)

        # Record legend labels to avoid replication
        if label is not None:
            self.legend[label] = style


class BarchartPlotter(Plotter):
    """Barchart plotter for generating direct comparison plots.

    :params figname: Name of output file
    :params plotdir: Directory to store the plot in
    :params title: Plot title to be printed on top

    Example usage:

    with BarchartPlotter(figname=..., plotdir=...) as barchart:
        barchart.add_point(gflops[0], oi[0], label='Point A')
        barchart.add_point(gflops[1], oi[1], label='Point B')
    """

    def __init__(self, figname='barchart', plotdir='plots',
                 title=None):
        super(BarchartPlotter, self).__init__(plotdir=plotdir)
        self.figname = figname
        self.title = title
        self.legend = {}
        self.values = defaultdict(dict)

    def __enter__(self):
        self.fig, self.ax = self.create_figure(self.figname)
        if self.title is not None:
            self.ax.set_title(title)
        return self

    def __exit__(self, *args):
        # Set axis labelling and generate plot file
        # self.ax.set_xticks(x_indices + width)
        # self.ax.set_xticklabels(self.values.keys())
        self.set_yaxis(self.ax, 'Runtime (s)')
        self.ax.legend(self.legend, loc='best', ncol=2,
                       fancybox=True, fontsize=10)
        self.save_figure(self.fig, self.figname)

    def add_value(self, value, grouplabel=None, color=None, label=None):
        """Adds a single point measurement to the barchart plot

        :param value: Y-value of the given point measurement
        :param grouplabel: Group label to be put on the X-axis
        :param color: Optional plotting color for data point
        :param label: Optional legend label for data point
        """
        # Record all points keyed by group and legend labels
        self.values[grouplabel][label] = value

        # Record legend labels to avoid replication
        if label is not None:
            self.legend[label] = color


class BarchartPlotter(Plotter):
    """Barchart plotter for generating direct comparison plots.

    :params figname: Name of output file
    :params plotdir: Directory to store the plot in
    :params title: Plot title to be printed on top

    Example usage:

    with BarchartPlotter(figname=..., plotdir=...) as barchart:
        barchart.add_point(gflops[0], oi[0], label='Point A')
        barchart.add_point(gflops[1], oi[1], label='Point B')
    """

    def __init__(self, figname='barchart', plotdir='plots',
                 title=None):
        super(BarchartPlotter, self).__init__(plotdir=plotdir)
        self.figname = figname
        self.title = title
        self.legend = {}
        self.values = defaultdict(dict)

    def __enter__(self):
        self.fig, self.ax = self.create_figure(self.figname)
        if self.title is not None:
            self.ax.set_title(title)
        return self

    def __exit__(self, *args):
        x_indices = np.arange(len(self.values)) + 0.1
        width = 0.8 / len(self.legend)
        for i, label in enumerate(self.legend):
            yvals = [val[label] for val in self.values.values()]
            lbl = self.ax.bar(x_indices + i * width, yvals, width,
                              color=self.legend[label])

        # Set axis labelling and generate plot file
        self.ax.set_xticks(x_indices + width)
        self.ax.set_xticklabels(self.values.keys())
        self.set_yaxis(self.ax, 'Runtime (s)')
        self.ax.legend(self.legend, loc='best', ncol=2,
                       fancybox=True, fontsize=10)
        self.save_figure(self.fig, self.figname)

    def add_value(self, value, grouplabel=None, color=None, label=None):
        """Adds a single point measurement to the barchart plot

        :param value: Y-value of the given point measurement
        :param grouplabel: Group label to be put on the X-axis
        :param color: Optional plotting color for data point
        :param label: Optional legend label for data point
        """
        # Record all points keyed by group and legend labels
        self.values[grouplabel][label] = value

        # Record legend labels to avoid replication
        if label is not None:
            self.legend[label] = color


class RooflinePlotter(Plotter):
    """Roofline plotter for generating generic roofline plots.

    :params figname: Name of output file
    :params plotdir: Directory to store the plot in
    :params title: Plot title to be printed on top
    :params max_bw: Maximum achievable memory bandwidth in GB/s.
                    This defines the slope of the roofline.
    :params max_flops: Maximum achievable performance in GFlops/s.
                       This defines the roof of the roofline.
    :params with_yminorticks: Show minor ticks on yaxis.
    :params fancycolors: Use beautiful colors, using the user-provided
                         colors as key to establish a 1-to-1 mapping
                         between user-provided colors and the new ones.
    :params legend: Additional arguments for legend entries, default:
                    {loc='best', ncol=2, fancybox=True, fontsize=10}

    Example usage:

    with RooflinePlotter(figname=..., plotdir=...,
                         max_bw=..., max_flops=...) as roofline:
        roofline.add_point(gflops[0], oi[0], label='Point A')
        roofline.add_point(gflops[1], oi[1], label='Point B')
    """

    def __init__(self, figname='roofline', plotdir='plots', title=None,
                 max_bw=None, max_flops=None, with_yminorticks=False,
                 fancycolor=False, legend=None):
        super(RooflinePlotter, self).__init__(plotdir=plotdir)
        self.figname = figname
        self.title = title
        self.legend = {'loc': 'best', 'ncol': 2, 'fancybox': True, 'fontsize': 10}
        self.legend.update(legend)  # Add user arguments to defaults
        self.legend_map = {}  # Label -> style map for legend entries

        self.max_bw = max_bw
        self.max_flops = max_flops
        self.xvals = [float(max_flops) / max_bw]
        self.yvals = [max_flops]
        self.with_yminorticks = with_yminorticks
        if fancycolor is True:
            self.fancycolor = ColorTracker({}, list(self.color))
        else:
            self.fancycolor = None

        # A set of OI values for which to add dotted lines
        self.oi_lines = []

    def __enter__(self):
        self.fig, self.ax = self.create_figure(self.figname)
        if self.title is not None:
            self.ax.set_title(self.title, {'fontsize': self.fonts['title']})
        return self

    def __exit__(self, *args):
        # Scale axis limits
        self.xvals = scale_limits(min(self.xvals), max(self.xvals), base=2., type='log')
        self.yvals = scale_limits(min(self.yvals), max(self.yvals), base=2., type='log')
        # Add a dotted lines for stored OI values
        for oi in self.oi_lines:
            self.ax.plot([oi, oi], [1., min(oi * self.max_bw, self.max_flops)], 'k:')

        # Add the roofline
        roofline = self.xvals * self.max_bw
        roofline[roofline > self.max_flops] = self.max_flops
        idx = (roofline >= self.max_flops).argmax()
        x_roofl = np.insert(self.xvals, idx, self.max_flops / self.max_bw)
        roofline = np.insert(roofline, idx, self.max_flops)
        self.ax.loglog(x_roofl, roofline, 'k-')

        # Set axis labelling and generate plot file
        xlabel='Operational intensity (Flops/Byte)'
        ylabel='Performance (GFlops/s)'
        self.set_xaxis(self.ax, xlabel, values=self.xvals, dtype=np.int32)
        self.set_yaxis(self.ax, ylabel, values=self.yvals, dtype=np.int32)
        self.ax.legend(**self.legend)
        self.save_figure(self.fig, self.figname)

    def set_yaxis(self, axis, label, values=None, dtype=np.float32):
        super(RooflinePlotter, self).set_yaxis(axis, label, values, dtype=np.float32)
        if values is not None:
            axis.yaxis.set_major_formatter(FormatStrFormatter("%d"))
            if self.with_yminorticks is True:
                axis.tick_params(axis='y', which='minor', labelsize=self.fonts['minorticks'])
                axis.yaxis.set_minor_formatter(FormatStrFormatter("%d"))
            else:
                axis.minorticks_off()

    def _select_point_color(self, usercolor):
        if usercolor is None:
            return self.color[0]
        elif not self.fancycolor:
            return usercolor
        elif usercolor not in self.fancycolor.mapper:
            try:
                fancycolor = self.fancycolor.available.pop(0)
                self.fancycolor.mapper[usercolor] = fancycolor
            except IndexError:
                bench_print("No more fancycolor available")
            return fancycolor
        else:
            return self.fancycolor.mapper[usercolor]

    def add_point(self, gflops, oi, marker=None, color=None, label=None, oi_line=True,
                  point_annotate=None, perf_annotate=None, oi_annotate=None):
        """Adds a single point measurement to the roofline plot

        :param gflops: Achieved performance in GFlops/s (y axis value)
        :param oi: Operational intensity in Flops/Byte (x axis value)
        :param marker: Optional plotting marker for point data
        :param color: Optional plotting color for point data
        :param label: Optional legend label for point data
        :param oi_line: Draw a vertical dotted line for the OI value
        :param point_annotate: Optional text to print next to point
        :param perf_annotate: Optional text showing the performance achieved
                              relative to the peak
        :param oi_annotate: Optional text or options dict to add an annotation
                            to the vertical OI line
        """
        self.xvals += [oi]
        self.yvals += [gflops]

        oi_top = min(oi * self.max_bw, self.max_flops)

        # Add dotted OI line and annotate
        if oi_line:
            self.ax.plot([oi, oi], [1., oi_top], ls=':', lw=0.3, c='black')
            if oi_annotate is not None:
                oi_ann = {'xy': (oi, 0.12), 'size': 8, 'rotation': -90,
                          'xycoords': ('data', 'axes fraction')}
                if isinstance(oi_annotate, Mapping):
                    oi_ann.update(oi_annotate)
                else:
                    oi_ann['s'] = oi_annotate
                plt.annotate(**oi_ann)

        # Add dotted gflops line
        if perf_annotate is not None:
            perf_ann = {'xy': (oi, oi_top), 'size': 5, 'textcoords': 'offset points',
                        'xytext': (-9, 4), 's': "%d%%" % (float("%.2f" % (gflops/oi_top))*100)}
            if isinstance(perf_annotate, Mapping):
                perf_ann.update(perf_annotate)
            plt.annotate(**perf_ann)

        # Plot and annotate the data point
        marker = marker or self.marker[0]
        self.ax.plot(oi, gflops, marker=marker, color=self._select_point_color(color),
                     label=label if label not in self.legend_map else None)
        if point_annotate is not None:
            p_ann = {'xy': (oi, gflops), 'size': 8, 'rotation': -45,
                     'xytext': (2, -13), 'textcoords': 'offset points',
                     'bbox': {'facecolor': 'w', 'edgecolor': 'none', 'pad': 1.0}}
            if isinstance(point_annotate, Mapping):
                p_ann.update(point_annotate)
            else:
                p_ann['s'] = point_annotate
            plt.annotate(**p_ann)

        # Record legend labels to avoid replication
        if label is not None:
            self.legend_map[label] = '%s%s' % (marker, color)


ColorTracker = namedtuple('ColorTracker', 'mapper available')
