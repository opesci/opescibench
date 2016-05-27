from argparse import ArgumentParser

__all__ = ['ArgParser']


class ArgParser(object):
    """ Wrapper for parsing standard benchmark arguments. """

    def __init__(self, **kwargs):
        self.parser = ArgumentParser(**kwargs)
        self.parser.add_argument('mode', choices=('bench', 'plot'), nargs='?', default='bench',
                                 help="Mode of operation; default: bench")
        self.parser.add_argument('-i', '--resultsdir', default='results',
                                 help='Directory containing results')

    def add_parameter(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def parse_args(self, *args, **kwargs):
        self.args = self.parser.parse_args(*args, **kwargs)
