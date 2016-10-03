#!/bin/bash

export PATH=~/.miniconda/bin:$PATH
export LIBRARY_PATH=~/.miniconda/lib:$LIBRARY_PATH
export LIBRARY_PATH=~/.miniconda/lib/python3.5/site-packages:$LIBRARY_PATH
export LIBRARY_PATH=~/.linuxbrew/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/vol/cuda/7.5.18/lib64:$LD_LIBRARY_PATH

scriptDir=$(dirname -- "$(readlink -e -- "$BASH_SOURCE")")
cd $scriptDir

python $1/run_local.py
