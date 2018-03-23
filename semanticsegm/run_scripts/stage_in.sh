#!/bin/bash

#arguments: 
#$1 - source dir
#$2 - destination dir
#$3 - how many files you shuffle

echo "Stage-in started"
echo "Creating directory"
mkdir -p ${2}
#echo "Get file list"
#FILELIST=$(ls -1  ${1}/data*| shuf -n ${3})
#echo "Copy"
#for f in $FILELIST; do
#    command="cp ${f} ${2}/"
#    #echo ${command}
#    ${command}
#done
#
#cp ${1}/stats.h5 ${2}/

echo "copy tar"
cp ${1}/ClimDeepLearn_data.tar ${2}

echo "untar"
cd ${2}
tar xf ClimDeepLearn_data.tar

echo "Stage-in done"

