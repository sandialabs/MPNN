#!/bin/bash -e

tar --exclude='*/__pycache__' --exclude='*/.DS_Store' -cvzf mpnn.tar examples fit mpnn postp prep setup.py build.sh README.md LICENSE