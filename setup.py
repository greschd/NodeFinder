#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from setuptools import setup, find_packages

readme = r"""TODO"""

with open('./nodefinder/_version.py', 'r') as f:
    match_expr = "__version__[^'" + '"]+([' + "'" + r'"])([^\1]+)\1'
    version = re.search(match_expr, f.read()).group(2).strip()

setup(
    name='nodefinder',
    version=version,
    author='Dominik Gresch, TODO',
    author_email='greschd@gmx.ch, TODO',
    description='TODO',
    install_requires=['numpy', 'scipy'],
    extras_require={'dev': ['pytest', 'yapf', 'pre-commit']},
    long_description=readme,
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English', 'Operating System :: Unix',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Development Status :: 4 - Beta'
    ],
    license='GPL',
    keywords=[],
    packages=find_packages()
)
