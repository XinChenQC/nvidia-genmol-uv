#!/bin/bash

SAFE_INIT=".venv/lib64/python3.12/site-packages/safe/__init__.py"

if [ ! -f "$SAFE_INIT" ]; then
    echo "Error: safe package not found at $SAFE_INIT"
    exit 1
fi

sed -i 's/^/# /' "$SAFE_INIT"
echo "from .converter import SAFEConverter, decode, encode" >> "$SAFE_INIT"

echo "Fixed safe package at $SAFE_INIT"

