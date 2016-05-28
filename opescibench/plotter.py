__all__ = ['Plotter']


class Plotter(object):
    """ Plotting utility that provides data and basic diagram utilities. """

    def __init__(self, benchmark, parser):
        self.bench = benchmark

        self.parser = parser
        self.parser.add_argument('--plottype', choices=('error', 'comparison'), default='error',
                                 help='Type of plot to generate: error-cost or barchart comparison')
        self.parser.add_argument('-i', '--resultsdir', default='results',
                                 help='Directory containing results')
        self.parser.add_argument('-o', '--plotdir', default='plots',
                                 help='Directory to store generated plots')
