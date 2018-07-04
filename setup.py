"""Usage: pip install .[dev]"""

import re
import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 5):
    raise 'Must use Python version 3.5 or higher.'

README = r"""TODO"""

with open('./nodefinder/__init__.py', 'r') as f:
    MATCH_EXPR = "__version__[^'\"]+(['\"])([^'\"]+)"
    VERSION = re.search(MATCH_EXPR, f.read()).group(2).strip()

EXTRAS_REQUIRE = dict(
    test=['pytest', 'pytest-cov', 'pytest-score'],
    doc=['sphinx', 'sphinx-rtd-theme'],
    dev=['pre-commit', 'prospector', 'yapf==0.20'],
)
EXTRAS_REQUIRE['dev'] += EXTRAS_REQUIRE['doc'] + EXTRAS_REQUIRE['test']

setup(
    name='nodefinder',
    version=VERSION,
    author='Dominik Gresch',
    author_email='greschd@gmx.ch',
    description='TODO',
    install_requires=[
        'numpy', 'scipy', 'matplotlib', 'decorator', 'fsc.export',
        'fsc.hdf5-io>=0.3.0', 'fsc.async_tools'
    ],
    extras_require=EXTRAS_REQUIRE,
    dependency_links=[
        'git+https://github.com/greschd/pytest-score.git@master#egg=pytest-score-0.0.0',
        'git+https://github.com/FrescolinoGroup/pyhdf5io.git@add_simple_base_class#egg=fsc.hdf5-io-0.3.0',
    ],
    long_description=README,
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
