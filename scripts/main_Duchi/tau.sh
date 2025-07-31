#!/bin/bash

mkdir -p log/main_Duchi

echo "lim-tau=60 ..."
{ time bash run.sh --lim-users=0 --lim-tau=60 --alg=main_Duchi ; } &>> log/main_Duchi/tau.log
echo "lim-tau=70 ..."
{ time bash run.sh --lim-users=0 --lim-tau=70 --alg=main_Duchi ; } &>> log/main_Duchi/tau.log
echo "lim-tau=80 ..."
{ time bash run.sh --lim-users=0 --lim-tau=80 --alg=main_Duchi ; } &>> log/main_Duchi/tau.log
echo "lim-tau=90 ..."
{ time bash run.sh --lim-users=0 --lim-tau=90 --alg=main_Duchi ; } &>> log/main_Duchi/tau.log
echo "lim-tau=100 ..."
{ time bash run.sh --lim-users=0 --lim-tau=100 --alg=main_Duchi ; } &>> log/main_Duchi/tau.log
echo "lim-tau=110 ..."
{ time bash run.sh --lim-users=0 --lim-tau=110 --alg=main_Duchi ; } &>> log/main_Duchi/tau.log
echo "lim-tau=0 ..."
{ time bash run.sh --lim-users=0 --lim-tau=0 --alg=main_Duchi ; } &>> log/main_Duchi/tau.log
