#!/bin/bash
set -e
echo "Setting up local editable environment..."
export UV_DYNAMIC_VERSIONING_BYPASS="2.2.0"
export PDM_BUILD_SCM_VERSION="2.2.0"
pip install -e ".[test]"
if [ -d "../ml-switcheroo-ir" ]; then
  pip install -e ../ml-switcheroo-ir
fi
if [ -d "../ml-ecosystem-snapshots" ]; then
  pip install -e ../ml-ecosystem-snapshots
fi
echo "Done!"
