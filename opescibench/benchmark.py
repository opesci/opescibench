from opescibench.plotter import Plotter
from opescibench.utils import bench_print, mpi_rank as rank

from argparse import ArgumentParser
from collections import OrderedDict, Iterable
from devito import configuration
from devito.mpi import MPI

from itertools import product
from datetime import datetime
from os import path, makedirs

import json
import mpi4py
from numpy import array


__all__ = ['Benchmark']


def get_argparsers(**kwargs):
    """ Utility that creates the root arguemnt parser and the sub-parser for each mode. """
    parser = ArgumentParser(**kwargs)
    subparsers = parser.add_subparsers(dest='mode', help="Mode of operation")
    parser_bench = subparsers.add_parser('bench', help='Perform benchmarking runs on target machine')
    parser_plot = subparsers.add_parser('plot', help='Plot diagrams from stored results')
    return parser, parser_bench, parser_plot


class Benchmark(object):
    """Class storing performance data for a set of benchmark runs
    indexed by a multi-parameter key.

    :param parameters: Dict of parameter names and value ranges that
                       defines the parameter space for this benchmark.
    :param resultsdir: Optional, define directory to store results in;
                       default: 'results'.
    :param name: Optional, set name of the benchmark instance;
                 default: 'Benchmark'.
    """

    def __init__(self, parameters, resultsdir='results', name='Benchmark'):
        self.name = name
        self.resultsdir = resultsdir

        self._params = parameters.keys()
        self._values = parameters

        self.timings = {}
        self.meta = {}

    @property
    def params(self):
        """ Lexicographically sorted parameter key """
        return tuple(sorted(self._params))

    @property
    def loaded(self):
        return self.timings and self.meta

    def values(self, keys=None):
        """ Sorted dict of parameter-value mappings for all parameters

        :param keys: Optional key-value dict to generate a subset of values
        """
        # Ensure all values are lists
        values = [(k, [v]) if not isinstance(v, list) else (k, v) for
                  k, v in self._values.items()]
        if keys is not None:
            # Ensure all keys are lists
            keys = dict([(k, [v]) if not isinstance(v, list) else (k, v)
                         for k, v in keys.items()])
            values = [(k, keys[k]) if k in keys else (k, v) for k, v in values]
        valdict = OrderedDict(sorted(values))
        assert(len(valdict) == len(self.params))
        return valdict

    def sweep(self, keys=None):
        """ List of value mappings for each instance of a parameter sweep.

        :param keys: Dict with parameter value mappings over which to sweep
        """
        values = self.values(keys=keys)
        return [OrderedDict(zip(self.params, v)) for v in product(*values.values())]

    def param_string(self, params):
        """ Convert parameter tuple to string """
        return '_'.join(['%s%s' % p for p in params])

    def lookup(self, params={}, event=None, measure='time', category='timings'):
        """ Lookup a set of results accoridng to a parameter set.

        :param params: Parameter set by which to filter results
        :param event: Optional, one or more events for which to retrieve data.
        :param category: Either 'timings' or 'meta'
        """
        assert(category in ['timings', 'meta'])
        result = OrderedDict()
        for params in self.sweep(params):
            key = tuple(params.items())
            datadict = getattr(self, category)
            if key in datadict:
                if event is None:
                    result[key] = datadict[key][measure]
                elif isinstance(event, Iterable) and not isinstance(event, str):
                    result[key] = [datadict[key][ev][measure]
                                   for ev in event]
                else:
                    result[key] = datadict[key][event][measure]
        return result

    def execute(self, executor, warmups=1, repeats=3):
        """
        Main execution function that invokes the given executor
        for each combination of the parameter sweep.
        """
        for params in self.sweep():
            bench_print("", pre=2)
            bench_print("Running %d repeats - parameters: %s" % (repeats,
                        ', '.join(['%s: %s' % (k, v) for k, v in params.items()])))

            # Execute the benchmark
            executor.execute(warmups=warmups, repeats=repeats, **params)

            # Store timing and meta data under the parameter key
            self.timings[tuple(params.items())] = executor.timings
            self.meta[tuple(params.items())] = executor.meta

            bench_print("", post=2)

    def save(self):
        """ Save all timing results in individually keyed files. """
        if rank > 0:
            return
        if not path.exists(self.resultsdir):
            makedirs(self.resultsdir)
        timestamp = datetime.now().strftime('%Y-%m-%dT%H%M%S')

        for key in self.timings.keys():
            datadict = OrderedDict()
            datadict['timestamp'] = timestamp
            datadict['meta'] = self.meta[key]
            datadict['timings'] = self.timings[key]

            if configuration['openmp']==1:
                cores_physical = configuration['platform'].cores_physical
            else:
                cores_physical = 0

            if configuration['mpi'] != 0:
                mpiprocs = MPI.COMM_WORLD.rank
            else:
                mpiprocs = 0

            filename = '%s_%s_mpi[%d]_nt[%d].json' % (self.name, self.param_string(key), mpiprocs, cores_physical)
            with open(path.join(self.resultsdir, filename), 'w') as f:
                json.dump(datadict, f, indent=4)



    def load(self):
        """ Load timing results from individually keyed files. """
        for params in self.sweep():
            filename = '%s_%s.json' % (self.name, self.param_string(params.items()))
            try:
                with open(path.join(self.resultsdir, filename), 'r') as f:
                    datadict = json.loads(f.read())
                    self.timings[tuple(params.items())] = datadict['timings']
                    self.meta[tuple(params.items())] = datadict['meta']
            except:
                bench_print("WARNING: Could not load file: %s" % filename)
