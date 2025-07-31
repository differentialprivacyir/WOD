#!/bin/bash

mkdir -p log/main_Duchi

echo "lim-dim=5 ..."
{ time bash run.sh --lim-users=0 --lim-dim=5 --alg=main_Duchi ; } &>> log/main_Duchi/dimensions.log
echo "lim-dim=7 ..."
{ time bash run.sh --lim-users=0 --lim-dim=7 --alg=main_Duchi ; } &>> log/main_Duchi/dimensions.log
echo "lim-dim=9 ..."
{ time bash run.sh --lim-users=0 --lim-dim=9 --alg=main_Duchi ; } &>> log/main_Duchi/dimensions.log
echo "lim-dim=11 ..."
{ time bash run.sh --lim-users=0 --lim-dim=11 --alg=main_Duchi ; } &>> log/main_Duchi/dimensions.log
echo "lim-dim=13 ..."
{ time bash run.sh --lim-users=0 --lim-dim=13 --alg=main_Duchi ; } &>> log/main_Duchi/dimensions.log
echo "lim-dim=0 ..."
{ time bash run.sh --lim-users=0 --lim-dim=0 --alg=main_Duchi ; } &>> log/main_Duchi/dimensions.log
