import numpy as np
import json as js
import re

from os import walk

from opescibench import *

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

figname = 'gflopss_omp'

atimings=([337.4,178.43,106.78,52.13,37.71])
agflopss=([13.27,25.08,42.15,85.84,118.67])
omp_threads=([1.0,2.0,4.0,8.0,12.0])
idealGF=np.asarray(omp_threads)/agflopss[0]
idealT=atimings[0]/np.asarray(omp_threads)


with LinePlotter(figname=figname, normalised=True) as plot:
        plot.add_line(omp_threads, idealGF , label='Ideal', style='g-^')
        plot.add_line(omp_threads, agflopss, label='Acoustic', style='b-s')
        #plot.add_line(omp_threads, gflopss, label='ViscoElastic', style='m-o')
        #plot.add_line(x3, y3, label='TTI', style='g-')
        #plot.ax.set_title('OpenMP GFlops/s scaling')
        plot.xlabel='Number of OpenMP threads'
        plot.ylabel='Speedup'


figname = 'timings_scaling'


with LinePlotter(figname=figname, normalised=False) as plot:
        #plot.add_line(omp_threads, idealT , label='Ideal', style='g-^')
        plot.add_line(omp_threads, atimings, label='Acoustic', style='b-s')
        #plot.add_line(omp_threads, gflopss, label='ViscoElastic', style='m-o')
        #plot.add_line(x3, y3, label='TTI', style='g-')
        #plot.ax.set_title('OpenMP execution time scaling')
        plot.xlabel='Number of OpenMP threads'
        plot.ylabel='Runtime(s)'
