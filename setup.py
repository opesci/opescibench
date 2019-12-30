try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='devitobench',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      description="Benchmarking tools for Devito Codes",
      long_descritpion="""devitobench is a set of performance
      benchmarking and plotting tools for simulation codes in the
      Devito project.""",
      url='http://www.devitoproject.org/devitobench',
      author="Imperial College London",
      author_email='g.gorman@imperial.ac.uk',
      license='MIT',
      packages=['devitobench'],
      install_requires=['numpy', 'simplejson'])
