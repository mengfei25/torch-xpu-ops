#!/bin/bash

if [ "$1" != "nightly_wheel" ];then
    source /home/gta/intel/oneapi/setvars.sh
else
    echo "Don't need to source DL-Essential for nightly wheel"
fi
