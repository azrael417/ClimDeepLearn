#!/bin/bash

#arguments: 
#$1 - source dir
#$2 - destination dir
#$3 - how many files you shuffle

echo "Stage-in started"
echo "Creating directory"
mkdir -p ${2}
echo "Get file list"
FILELIST=$(ls -1  ${1}/data*| shuf -n ${3})
echo "Copy"
for f in $FILELIST; do
    command="cp ${f} ${2}/"
    echo ${command}
    ${command}
done
echo "Stage-in done"

