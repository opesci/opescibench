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
        yvals.append(res_dict['gflopss']['section0'][var])
    if _ordered:
        xvals, yvals = (list(i) for i in zip(*sorted(zip(xvals, yvals))))
    return xvals, yvals


# Rank of interest
rank = 'rank[0]'

path = '/home/gb4018/opesci/devito/benchmarks/user/results/elastic/'
files = scan_dir(path)
files = [i for i in files if rank in i]
x0, y0 = extract_xy(files, path)

path = '/home/gb4018/opesci/devito/benchmarks/user/results/acoustic/'
files = scan_dir(path)
files = [i for i in files if rank in i]
x1, y1 = extract_xy(files, path)

path = '/home/gb4018/opesci/devito/benchmarks/user/results/viscoelastic/'
files = scan_dir(path)
files = [i for i in files if rank in i]
x2, y2 = extract_xy(files, path)

path = '/home/gb4018/opesci/devito/benchmarks/user/results/tti/'
files = scan_dir(path)
files = [i for i in files if rank in i]
x3, y3 = extract_xy(files, path)


figname = 'test'

with LinePlotter(figname=figname, normalised=True) as plot:
        #plot.add_line(x0, y0, label='Elastic')
        plot.add_line(x1, y1, label='Acoustic', style='r-o')
        plot.add_line(x2, y2, label='ViscoElastic', style='m-o')
        #plot.add_line(x3, y3, label='TTI', style='g-')
