#!/bin/bash

if [[ $# == 1 ]]; then
    docker run -it -e LIM=$1 --rm wheel-of-differential:0.1.0
else
    docker run -it --rm wheel-of-differential:0.1.0
fi
