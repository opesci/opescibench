try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='opescibench',
      version='0.0.2',
      description="Benchmarking tools for OPESCI codes",
      long_descritpion="""Opescibench is a set of performance
      benchmarking and plotting tools for simulation codes in the
      OPESCI project.""",
      url='http://www.opesci.org/opescibench',
      author="Imperial College London",
      author_email='opesci@imperial.ac.uk',
      license='MIT',
      packages=['opescibench'],
      install_requires=['numpy', 'simplejson'])
