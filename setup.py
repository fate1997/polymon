#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

from polymon import __version__

setup(
    author="PolyMon Group",
    description="PolyMon",
    name='polymon',
    packages=find_packages(include=['polymon', 'polymon.*', 'polymon.*.*']),
    package_data={'': ['*.yml', '*.txt']},
    include_package_data=True,
    version=__version__,
)