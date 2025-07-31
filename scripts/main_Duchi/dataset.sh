#!/bin/bash

mkdir -p log/main_Duchi

echo "dat-num=2 ..."
{ time bash run.sh --lim-users=0 --dat-num=2 --alg=main_Duchi ; } &>> log/main_Duchi/dataset.log
echo "dat-num=4 ..."
{ time bash run.sh --lim-users=0 --dat-num=4 --alg=main_Duchi ; } &>> log/main_Duchi/dataset.log
echo "dat-num=7 ..."
{ time bash run.sh --lim-users=0 --dat-num=7 --alg=main_Duchi ; } &>> log/main_Duchi/dataset.log
echo "dat-num=8 ..."
{ time bash run.sh --lim-users=0 --dat-num=8 --alg=main_Duchi ; } &>> log/main_Duchi/dataset.log
echo "dat-num=9 ..."
{ time bash run.sh --lim-users=0 --dat-num=9 --alg=main_Duchi ; } &>> log/main_Duchi/dataset.log
echo "dat-num=10 ..."
{ time bash run.sh --lim-users=0 --dat-num=10 --alg=main_Duchi ; } &>> log/main_Duchi/dataset.log
echo "dat-num=11 ..."
{ time bash run.sh --lim-users=0 --dat-num=11 --alg=main_Duchi ; } &>> log/main_Duchi/dataset.log
echo "dat-num=14 ..."
{ time bash run.sh --lim-users=0 --dat-num=14 --alg=main_Duchi ; } &>> log/main_Duchi/dataset.log
