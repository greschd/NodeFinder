# -*- coding: utf-8 -*-

# © 2017-2019, ETH Zurich, Institut für Theoretische Physik
# Author: Dominik Gresch <greschd@gmx.ch>
"""Usage: pip install .[dev]"""

import re
import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 5):
    raise 'Must use Python version 3.5 or higher.'

with open('./README.md', 'r') as f:
    README_MD = f.read()
try:
    import pypandoc
    README = pypandoc.convert_text(README_MD, 'rst', 'md')
except:
    README = ''

with open('./nodefinder/__init__.py', 'r') as f:
    MATCH_EXPR = "__version__[^'\"]+(['\"])([^'\"]+)"
    VERSION = re.search(MATCH_EXPR, f.read()).group(2).strip()

EXTRAS_REQUIRE = dict(
    test=['pytest', 'pytest-cov', 'pytest-score'],
    doc=['sphinx', 'sphinx-rtd-theme', 'ipython>=6.2'],
    dev=[
        'pylint==2.1.1', 'pre-commit==1.11.1', 'prospector==1.1.2',
        'yapf==0.24', 'pypandoc'
    ],
)
EXTRAS_REQUIRE['dev'] += EXTRAS_REQUIRE['doc'] + EXTRAS_REQUIRE['test']

setup(
    name='nodefinder',
    version=VERSION,
    author='Dominik Gresch',
    author_email='greschd@gmx.ch',
    description=
    'A tool for studying the nodal features of potential lanscapes.',
    install_requires=[
        'numpy', 'scipy', 'matplotlib', 'decorator', 'fsc.export',
        'fsc.hdf5-io>=0.4.0', 'fsc.async_tools', 'networkx>=2.0'
    ],
    extras_require=EXTRAS_REQUIRE,
    dependency_links=[
        'git+https://github.com/greschd/pytest-score.git@master#egg=pytest-score-0.0.0',
    ],
    long_description=README,
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English', 'Operating System :: Unix',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Development Status :: 4 - Beta'
    ],
    license='Apache 2.0',
    keywords=['minimization', 'node search'],
    packages=find_packages()
)
