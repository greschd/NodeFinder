#!/bin/bash

# Author: Dominik Gresch <greschd@gmx.ch>

# Be verbose, and stop with error as soon there's one
set -ev

cd ${TRAVIS_BUILD_DIR}

pip install git+https://github.com/greschd/pytest-score.git

case "$INSTALL_TYPE" in
    dev)
        pip install .[dev]
        ;;
    test)
        pip install .[test]
        ;;
    test_sdist)
        python setup.py sdist
        ls -1 dist/ | xargs -I % pip install dist/%[test]
        ;;
esac
