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

def extract_xy(files, path, _ordered=True):
    # Get 'xvals' from filenames:
    xvals = []
    yvals = []
    for f in files:
        xvals.append(int(re.findall(r'\d+', f)[-2]))
        with open(path+f, 'r') as f:
            res_dict = js.load(f)
        yvals.append(res_dict['timings']['section0']['gflopss'])
    if _ordered:
        xvals, yvals = (list(i) for i in zip(*sorted(zip(xvals, yvals))))
    return xvals, yvals

path = '/data/cx2_data/benchmarks/elastic/results/'
files = scan_dir(path)

rank = 'rank[0]'
files = [i for i in files if rank in i]

x0, y0 = extract_xy(files, path)

path = '/data/cx2_data/benchmarks/viscoelastic/results/'
files = scan_dir(path)

rank = 'rank[0]'
files = [i for i in files if rank in i]

x1, y1 = extract_xy(files, path)

figname = 'test'

with LinePlotter(figname=figname) as plot:
        plot.add_line(x0, y0, label='Elastic')
        plot.add_line(x1, y1, label='Viscolastic', style='r-')

#from IPython import embed; embed()

##############################################################
    
    #fdict = {}
    #model = opts['model']
    #arch = opts['arch']
    #shape = opts['shape']
    #nbpml = opts['nbpml']
    #tn = opts['tn']
    #so = opts['so']
    #to = opts['to']
    #dse = opts['dse']
    #dle = opts['dle']
    #at = opts['at']
    #nt = opts['nt']
    ## These options need to be sub-'dictioneried'
    #np = opts['np']
    #rank = opts['rank']
    
    #fname = 


#class Test(object):
    #def __init__(self, data):
	    #self.__dict__ = json.loads(data)
#test1 = Test(json_data)
#print(test1.a)

#elastic_arch[unknown]_shape[512,512,512]_nbpml[10]_tn[250]_so[2]_to[2]_dse[advanced]_dle[advanced]_at[aggressive]_nt[24]_np[4]_rank[0]

#with open('distros.json', 'r') as f:
    #distros_dict = js.load(f)

#from IPython import embed; embed()

#for distro in distros_dict:
    #print(distro['Name'])

#xvals0 = [1., 2., 3., 4., 5.]
#yvals0 = [0.5, 0.9, 1.42, 1.87, 2.3]

#xvals1 = [1., 2., 3., 4., 5.]
#yvals1 = [0.6, 1.1, 1.5, 1.9, 2.4]

#figname = 'test'

#with LinePlotter(figname=figname) as plot:
        #plot.add_line(xvals0, yvals0)
        #plot.add_line(xvals1, yvals1, label='Strong scaling')
