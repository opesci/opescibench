try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='opescibench',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
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
