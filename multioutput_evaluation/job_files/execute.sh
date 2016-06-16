#!/bin/bash

export PATH=''
export CPATH=''
export LIBRARY_PATH=''
#export LD_LIBRARY_PATH=''

SOURCES=($HOME/.miniconda $HOME/.linuxbrew /usr/local /usr /)
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
python run_local.py
