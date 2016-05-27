from opescibench.argparser import ArgParser
from collections import OrderedDict
from itertools import product
from datetime import datetime
from os import path
import json

__all__ = ['Benchmark']


class Benchmark(object):
    """ Class encapsulating a set of benchmark runs. """

    def __init__(self, name, **kwargs):
        self.name = name
        self.parser = ArgParser(**kwargs)

        self._params = []
        self._values = {}

        self.timings = {}
        self.meta = {}

    def add_parameter(self, name, *parserargs, **parserkwargs):
        self._params += [name.lstrip('-')]
        self.parser.add_parameter(name, *parserargs, **parserkwargs)

    def add_parameter_value(self, name, values):
        self._params += [name]
        self._values[name] = values

    @property
    def params(self):
        """ Lexicographically sorted parameter key """
        return tuple(sorted(self._params))

    @property
    def values(self):
        """ Sorted dict of parameter-value mappings """
        for p in self.params:
            if hasattr(self.parser.args, p):
                self._values[p] = getattr(self.parser.args, p)
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
        resultsdir = self.parser.args.resultsdir
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
        resultsdir = self.parser.args.resultsdir
        for params in self.sweep:
            filename = '%s_%s.json' % (self.name, self.param_key(params.items()))
            try:
                with open(path.join(resultsdir, filename), 'r') as f:
                    self.timings[tuple(params.items())] = json.loads(f.read())
            except:
                print "WARNING: Could not load file: %s" % filename
