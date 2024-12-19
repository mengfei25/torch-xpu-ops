#!/bin/bash

if [ "$1" != "nightly_wheel" ];then
    source ${HOME}/intel/oneapi/compiler/latest/env/vars.sh
    source ${HOME}/intel/oneapi/umf/latest/env/vars.sh
    source ${HOME}/intel/oneapi/pti/latest/env/vars.sh
else
    echo "Don't need to source DL-Essential for nightly wheel"
fi
