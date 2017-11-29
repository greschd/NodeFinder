#!/bin/bash
py.test -p no:cov-exclude --cov=nodefinder --cov-report=html
