import numpy as np
import json as js
import re

from os import walk

from opescibench import LinePlotter

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
        yvals.append(res_dict['timings']['section0'][var])
    if _ordered:
        xvals, yvals = (list(i) for i in zip(*sorted(zip(xvals, yvals))))
    return xvals, yvals


# Rank of interest
rank = 'rank[0]'

path = '/data/cx2_data/benchmarks/elastic/results/'
files = scan_dir(path)
files = [i for i in files if rank in i]
x0, y0 = extract_xy(files, path)

path = '/data/cx2_data/benchmarks/viscoelastic/results/'
files = scan_dir(path)
files = [i for i in files if rank in i]
x1, y1 = extract_xy(files, path)

path = '/data/cx2_data/benchmarks/acoustic/results/'
files = scan_dir(path)
files = [i for i in files if rank in i]
x2, y2 = extract_xy(files, path)

path = '/data/cx2_data/benchmarks/tti/results/'
files = scan_dir(path)
files = [i for i in files if rank in i]
x3, y3 = extract_xy(files, path)


figname = 'test'

with LinePlotter(figname=figname, normalised=True) as plot:
        plot.add_line(x0, y0, label='Elastic')
        plot.add_line(x1, y1, label='Acoustic', style='r-')
        plot.add_line(x2, y2, label='ViscoElastic', style='m-')
        plot.add_line(x3, y3, label='TTI', style='g-')
