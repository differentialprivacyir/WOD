#!/bin/bash

# --- Function to display help message ---
show_help() {
  echo "Usage: ./run.sh [OPTIONS]"
  echo "A script to run wheel of differential."
  echo
  echo "Options:"
  echo "  --eps=1           Set the epsilon to 1 (default is 1)."
  echo "  --dat-num=2       Set the dataset number to 2 (default is 1)."
  echo "                    Dataset number can be one of these:"
  echo "                      [2, 4, 7, 8, 9, 10, 11, 14]"
  echo "  --lim-dim=0       Set the limited dimensions to 0 (default is not-limited)."
  echo "  --lim-tau=0       Set the limited tau to 0 (default is not-limited)."
  echo "  --lim-users=100   Set the limited users to 100 (default is 100)."
  echo "  --alg=main        Set the algorithm to main (default is main)."
  echo "                    Algorithm can be one of these:"
  echo "                      - main"
  echo "                      - main_Rappor"
  echo "                      - main_PM"
  echo "                      - main_Duchi"
  echo "                      - main_dBitFlipPM"
  echo "  -h, --help        Display this help message and exit."
}

# Initialize variables 
EPSILON=1
DATASET_NUMBER=2
LIMITED_DIMENSIONS=0
LIMITED_TAU=0
LIMITED_NUMBER=100

ALGORITHM=""

# --- Process Command-Line Arguments --- 
while (( "$#" )); do 
  case "$1" in 
    --eps=*) 
      EPSILON="${1#*=}" 
      shift 
      ;; 
    --dat-num=*) 
      DATASET_NUMBER="${1#*=}" 
      shift 
      ;; 
    --lim-dim=*) 
      LIMITED_DIMENSIONS="${1#*=}" 
      shift 
      ;; 
    --lim-tau=*) 
      LIMITED_TAU="${1#*=}" 
      shift 
      ;; 
    --lim-users=*) 
      LIMITED_NUMBER="${1#*=}" 
      shift 
      ;; 
    --alg=*) 
      ALGORITHM="${1#*=}" 
      shift 
      ;; 
    -h|--help)
      show_help
      exit 0
      ;;
    *) 
      echo "Invalid option: $1" >&2 
      exit 1 
      ;; 
  esac 
done 

if [ -n "$ALGORITHM" ]; then 
  ALGORITHM="python $ALGORITHM.py" 
fi 

docker run -it --rm \
    -e LIM=$LIMITED_NUMBER \
    -e LIM_tAU=$LIMITED_TAU \
    -e LIM_DIM=$LIMITED_DIMENSIONS \
    -e DATASET_NUMBER=$DATASET_NUMBER \
    -e EPSILON=$EPSILON \
    wheel-of-differential:0.1.0 $ALGORITHM
