#!/usr/bin/env python
from setuptools import setup
from setuptools import find_packages

setup(
    name='magic',
    version='0.0.1',
    description='Author Profiler Designed for PAN 2016',
    url='http://dbs.cs.uni-duesseldorf.de',
    author='HHU Duesseldorf',
    author_email='modaresi@cs.uni-duesseldorf.de',
    packages=find_packages(exclude=['tests']),
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 2.7',
    ],
    install_requires=[]
)
