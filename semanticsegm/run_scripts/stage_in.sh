#!/bin/bash

echo "Creating directory"
mkdir -p ${2}
echo "Copying data"
#cp -r ${1}/* ${2}/
for x in $(ls ${1} | head -n 20); do
    cp -r ${1}/${x} ${2}/
done
