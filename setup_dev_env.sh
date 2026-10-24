#!/bin/bash
set -e
echo "Setting up local editable environment..."
pip install -e ".[test]"
if [ -d "../ml-switcheroo-ir" ]; then
  pip install -e ../ml-switcheroo-ir
fi
echo "Done!"
