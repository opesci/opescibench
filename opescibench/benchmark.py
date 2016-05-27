from opescibench.argparser import ArgParser

__all__ = ['Benchmark']


class Benchmark(object):
    """ Class encapsulating a set of benchmark runs. """

    def __init__(self, name, **kwargs):
        self.parser = ArgParser(**kwargs)

        self._params = []
        self._paramvalues = []

    def add_parameter(self, name, *parserargs, **parserkwargs):
        self._params += [name.lstrip('-')]
        self.parser.add_parameter(name, *parserargs, **parserkwargs)

    def add_parameter_values(self, name, values):
        self._params += [name]
        self._paramvalues += [values]

    @property
    def params(self):
        return tuple(sorted(self._params))
