import numpy as np
import json as js
import re

from os import walk

from opescibench import *
import matplotlib.ticker as mtick


def scan_dir(path='.', include=None, exclude=None, _extension='json'):
    (_, _, filenames) = walk(path).__next__()
    if _extension:
        filenames = [i for i in filenames if '.json' in i]
    if include:
        if not isinstance(include, str):
            raise TypeError("include must be a string")
        filenames = [i for i in filenames if include in i]
    if exclude:
        if not isinstance(exclude, str):
            raise TypeError("exclude must be a string")
        filenames = [i for i in filenames if include in i]
    return filenames


def extract_xy(files, path, var='gflopss', _ordered=True):
    # Get 'vals' from filenames:
    xvals = []
    yvals = []
    for f in files:
        xvals.append(int(re.findall(r'\d+', f)[-2]))
        with open(path+f, 'r') as f:
            res_dict = js.load(f)
        yvals.append(res_dict['gflopss']['section0'][var])
    if _ordered:
        xvals, yvals = (list(i) for i in zip(*sorted(zip(xvals, yvals))))
    return xvals, yvals


figname = 'speedup_omp'

atimings = ([337.4, 178.43, 106.78, 52.13, 37.71])
vtimings = ([907.01, 689.14, 360.12, 183.70, 124.11])

agflopss = ([13.27, 25.08, 42.15, 85.84, 118.67])
vgflopss = ([3.99, 5.25, 10.04, 19.67, 29.11])

omp_threads = ([1.0, 2.0, 4.0, 8.0, 12.0])
idealGF = np.asarray(omp_threads)/vgflopss[0]
idealT = atimings[0]/np.asarray(omp_threads)


with LinePlotter(figname=figname, normalised=True) as plot:
    plot.add_line(omp_threads, 1/idealT, label='Ideal', style='g-^')
    plot.add_line(omp_threads, atimings[0]/np.asarray(atimings), label='Acoustic', style='b-s')
    plot.add_line(omp_threads, vtimings[0]/np.asarray(vtimings), label='ViscoElastic', style='r-o')
    #plot.add_line(x3, y3, label='TTI', style='g-')
    #plot.ax.set_title('OpenMP GFlops/s scaling')
    plot.xlabel = 'Number of OpenMP threads'
    plot.ylabel = 'Speedup'
    plot.ax.grid()
    plot.ax.set_xscale('log', basex=2)
    plot.ax.set_yscale('log', basey=2)

    plot.ax.set_xticks(omp_threads)
    plot.ax.set_xticklabels(omp_threads)

    plot.ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.f'))
    plot.ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.f'))

figname = 'timings_scaling'


with LinePlotter(figname=figname, normalised=False) as plot:
    # plot.add_line(omp_threads, idealT, label='Ideal', style='g-^')
    plot.add_line(omp_threads, atimings, label='Acoustic', style='b-s')
    plot.add_line(omp_threads, vtimings, label='ViscoElastic', style='r-o')
    #plot.add_line(x3, y3, label='TTI', style='g-')
    #plot.ax.set_title('OpenMP execution time scaling')
    plot.ax.grid()



    plot.ax.set_xscale('log', basex=2)
    plot.ax.set_yscale('log', basey=2)


    plot.ax.set_xticks(omp_threads)
    plot.ax.set_xticklabels(omp_threads)
    plot.ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.f'))
    plot.ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.f'))

    plot.xlabel = 'Number of OpenMP threads'
    plot.ylabel = 'Runtime(s)'
