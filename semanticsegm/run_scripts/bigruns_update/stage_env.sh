#!/bin/bash
# Copy and extract python environment (requires spectrum-mpi and cuda modules to be loaded)
if [ ! -d "${2}" ]; then
    mkdir -p ${2}
fi
cp /gpfs/alpinetds/world-shared/ven201/seant/climate/pyvenv_summit_7.5.18.tar ${2}
tar xf ${2}/pyvenv_summit_7.5.18.tar -C ${2}
