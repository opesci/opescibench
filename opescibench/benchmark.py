from opescibench.argparser import ArgParser
from collections import OrderedDict
from itertools import product

__all__ = ['Benchmark']


class Benchmark(object):
    """ Class encapsulating a set of benchmark runs. """

    def __init__(self, name, **kwargs):
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
