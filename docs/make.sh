#!/bin/bash
sphinx-apidoc -f -o source/ ../tools/
sphinx-apidoc -f -o source/ ../tests/
sphinx-apidoc -f -o source/ ../examples/
make html
