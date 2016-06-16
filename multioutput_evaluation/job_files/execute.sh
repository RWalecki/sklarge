#!/bin/bash

export PATH=''
export CPATH=''
export LIBRARY_PATH=''
#export LD_LIBRARY_PATH=''
#PYTHONPATH=$PYTHONPATH:/homes/rw2614/.miniconda/lib/python3.5/site-packages/

SOURCES=(/homes/rw2614/.miniconda /homes/rw2614/.linuxbrew /usr/local /usr /)
for i in ${SOURCES[*]}; do

    export PATH=$PATH:$i/bin
    export PATH=$PATH:$i/sbin
    export CPATH=$CPATH:$i/include
    export LIBRARY_PATH=$LIBRARY_PATH:$i/lib
    #export LD_LIBRARY_PATH=$LD_LIBRARY_PATH$i/lib:
done

export PYTHONWARNINGS="ignore"
export PATH=$PATH:/Developer/NVIDIA/CUDA-7.5/bin

scriptDir=$(dirname -- "$(readlink -e -- "$BASH_SOURCE")")
cd $scriptDir
python $1/run_local.py
