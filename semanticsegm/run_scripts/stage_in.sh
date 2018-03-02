#!/bin/bash

echo "Creating directory"
mkdir -p ${2}
echo "Copying data"
cp -r ${1}/* ${2}/
