from collections import defaultdict

__all__ = ['Executor']


class Executor(object):
    """ Abstract container class for a single benchmark data point. """

    def __init__(self):
        self.meta = {}
        self.timings = defaultdict(lambda: defaultdict(float))

    def setup(self, **kwargs):
        """ Prepares a single benchmark invocation. """
        pass

    def teardown(self, **kwargs):
        """ Cleans up a single benchmark invocation. """
        pass

    def postprocess(self, **kwargs):
        """ Global post-processing method to collect meta-data. """
        pass

    def run(self, **kwargs):
        """ This methods needs to be overridden byt the user. """
        raise NotImplementedError("No custom executor function specified")

    def register(self, event, measure, value):
        """
        Register a single timing value for a given event key.

        :param event: key for the measured event, ie. 'assembly' or 'solve'
        :param measure: name of the value type, eg. 'time' or 'flops'
        :param value: measured value to store

        """
        self.timings[event][measure] += value

    def execute(self, warmups=1, repeats=3, **params):
        """
        Execute a single benchmark repeatedly, including
        setup, teardown and postprocessing methods.
        """
        for _ in range(warmups):
            self.setup(**params)
            self.run(**params)
            self.teardown(**params)

        for _ in range(repeats):
            self.setup(**params)
            self.run(**params)
            self.teardown(**params)

            # Average timings across repeats
            for event in self.timings.keys():
                for measure in self.timings[event].keys():
                    self.timings[event][measure] /= repeats

        # Collect meta-information via post-processing methods
        self.postprocess(**params)
