from opescibench.plotter import Plotter
from argparse import ArgumentParser
from collections import OrderedDict
from itertools import product
from datetime import datetime
from os import path
import json

__all__ = ['Benchmark']


def get_argparsers(**kwargs):
    """ Utility that creates the root arguemnt parser and the sub-parser for each mode. """
    parser = ArgumentParser(**kwargs)
    subparsers = parser.add_subparsers(dest='mode', help="Mode of operation")
    parser_bench = subparsers.add_parser('bench', help='Perform benchmarking runs on target machine')
    parser_plot = subparsers.add_parser('plot', help='Plot diagrams from stored results')
    return parser, parser_bench, parser_plot


class Benchmark(object):
    """ Class encapsulating a set of benchmark runs. """

    def __init__(self, name, **kwargs):
        self.name = name
        # Initialise root arguemnt parser and bench/plot subparsers
        self._root_parser, self.parser, plot_parser = get_argparsers(**kwargs)
        self.parser.add_argument('-i', '--resultsdir', default='results',
                                 help='Directory containing results')

        # Create plotter with a dedicated subparser
        self.plotter = Plotter(self, plot_parser, **kwargs)

        self._params = []
        self._values = {}

        self.timings = {}
        self.meta = {}

    def add_parameter(self, name, *parserargs, **parserkwargs):
        self._params += [name.lstrip('-')]
        self.parser.add_argument(name, *parserargs, **parserkwargs)
        self.plotter.parser.add_argument(name, *parserargs, **parserkwargs)

    def add_parameter_value(self, name, values):
        self._params += [name]
        self._values[name] = values

    def parse_args(self, **kwargs):
        """ Parse arguments using the root parser. """
        self.args = self._root_parser.parse_args(**kwargs)

    @property
    def params(self):
        """ Lexicographically sorted parameter key """
        return tuple(sorted(self._params))

    @property
    def values(self):
        """ Sorted dict of parameter-value mappings """
        for p in self.params:
            if hasattr(self.args, p):
                self._values[p] = getattr(self.args, p)
        # Ensure all values are lists
        valuelist = [(k, [v]) if not isinstance(v, list) else (k, v)
                     for k, v in self._values.items()]
        return OrderedDict(sorted(valuelist))

    @property
    def sweep(self):
        """ List of value mappings for each instance of a parameter sweep. """
        return [OrderedDict(zip(self.params, v)) for v in product(*self.values.values())]

    def param_key(self, params):
        """ Convert parameter tuple to string """
        return '_'.join(['%s%s' % p for p in params])

    def lookup(self, event, measure, paramset):
        """ Lookup a set of results accoridng to a parameter set. """
        print "Lookup::%s::%s: %s" % (event, measure, paramset)

    def execute(self, executor, warmups=1, repeats=3):
        """
        Main execution function that invokes the given executor
        for each combination of the parameter sweep.
        """
        for params in self.sweep:
            # Execute the benchmark
            executor.execute(warmups=warmups, repeats=repeats, **params)

            # Store timing and meta data under the parameter key
            self.timings[tuple(params.items())] = executor.timings
            self.meta[tuple(params.items())] = executor.meta

    def save(self):
        """ Save all timing results in individually keyed files. """
        resultsdir = self.args.resultsdir
        timestamp = datetime.now().strftime('%Y-%m-%dT%H%M%S')

        for key in self.timings.keys():
            outdict = OrderedDict()
            outdict['timestamp'] = timestamp
            outdict['meta'] = self.meta[key]
            outdict['timings'] = self.timings[key]

            filename = '%s_%s.json' % (self.name, self.param_key(key))
            with open(path.join(resultsdir, filename), 'w') as f:
                json.dump(outdict, f, indent=4)

    def load(self):
        """ Load timing results from individually keyed files. """
        resultsdir = self.args.resultsdir
        for params in self.sweep:
            filename = '%s_%s.json' % (self.name, self.param_key(params.items()))
            try:
                with open(path.join(resultsdir, filename), 'r') as f:
                    self.timings[tuple(params.items())] = json.loads(f.read())
            except:
                print "WARNING: Could not load file: %s" % filename
