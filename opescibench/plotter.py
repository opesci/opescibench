__all__ = ['Plotter']

class Plotter(object):
    """ Plotting utility that provides data and basic diagram utilities. """

    def __init__(self, benchmark, parser):
        self.bench = benchmark

        self.parser = parser
        self.parser.add_argument('--plottype', choices=('error', 'comparison'),
                                  help='Type of plot to generate')
        self.parser.add_argument('-i', '--resultsdir', default='results',
                                 help='Directory containing results')
        self.parser.add_argument('-o', '--plotdir', default='plots',
                                 help='Directory to store generated plots')

