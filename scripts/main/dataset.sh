#!/bin/bash

mkdir -p log/main

echo "dat-num=2 ..."
{ time bash run.sh --lim-users=0 --dat-num=2 ; } &>> log/main/dataset.log
echo "dat-num=4 ..."
{ time bash run.sh --lim-users=0 --dat-num=4 ; } &>> log/main/dataset.log
echo "dat-num=7 ..."
{ time bash run.sh --lim-users=0 --dat-num=7 ; } &>> log/main/dataset.log
echo "dat-num=8 ..."
{ time bash run.sh --lim-users=0 --dat-num=8 ; } &>> log/main/dataset.log
echo "dat-num=9 ..."
{ time bash run.sh --lim-users=0 --dat-num=9 ; } &>> log/main/dataset.log
echo "dat-num=10 ..."
{ time bash run.sh --lim-users=0 --dat-num=10 ; } &>> log/main/dataset.log
echo "dat-num=11 ..."
{ time bash run.sh --lim-users=0 --dat-num=11 ; } &>> log/main/dataset.log
echo "dat-num=14 ..."
{ time bash run.sh --lim-users=0 --dat-num=14 ; } &>> log/main/dataset.log
