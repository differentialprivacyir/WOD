#!/bin/bash

mkdir -p log/main

echo "lim-dim=5 ..."
{ time bash run.sh --lim-users=0 --lim-dim=5 ; } &>> log/main/dimensions.log
echo "lim-dim=7 ..."
{ time bash run.sh --lim-users=0 --lim-dim=7 ; } &>> log/main/dimensions.log
echo "lim-dim=9 ..."
{ time bash run.sh --lim-users=0 --lim-dim=9 ; } &>> log/main/dimensions.log
echo "lim-dim=11 ..."
{ time bash run.sh --lim-users=0 --lim-dim=11 ; } &>> log/main/dimensions.log
echo "lim-dim=13 ..."
{ time bash run.sh --lim-users=0 --lim-dim=13 ; } &>> log/main/dimensions.log
echo "lim-dim=0 ..."
{ time bash run.sh --lim-users=0 --lim-dim=0 ; } &>> log/main/dimensions.log
