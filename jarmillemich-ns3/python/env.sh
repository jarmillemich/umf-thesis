#!/usr/bin/env bash

NS3=`pwd`/../../ns-3-dev/build

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NS3/lib:$NS3
export NS3_MODULE_PATH=$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:$NS3/bindings/python