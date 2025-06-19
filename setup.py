#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

setup(
    author="PolyMon Group",
    description="PolyMon",
    name='polymon',
    packages=find_packages(include=['polymon', 'polymon.*', 'polymon.*.*']),
    package_data={'': ['*.yml']},
    install_requires=[
        'torch',
        'pandas',
        'scikit-learn',
        'torchmetrics'
    ],
    include_package_data=True,
    version='0.1.0',
)