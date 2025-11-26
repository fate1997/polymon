#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

from polymon import __version__

setup(
    author="PolyMon Group",
    description="PolyMon",
    name='polymon',
    packages=find_packages(include=['polymon', 'polymon.*', 'polymon.*.*']),
    package_data={'': ['*.yml', '*.txt', '*.csv', '*.json', '*.npy']},
    include_package_data=True,
    version=__version__,
    entry_points={
        'console_scripts': [
            'polymon = polymon.cli.main:main',
        ],
    },
    install_requires=[
        "mordredcommunity",
        "mordred==1.2.0",
        "rdkit==2023.09.6",
        "xenonpy",
        "xgboost",
        "catboost",
        "lightgbm",
        "loguru",
        "scikit-learn",
        "tabpfn",
        "torchensemble",
        "optuna",
        "lightning",
        "torch_geometric",
        "pykan"
    ],
)