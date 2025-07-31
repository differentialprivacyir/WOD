#!/bin/bash

mkdir -p log/main

echo "lim-tau=60 ..."
{ time bash run.sh --lim-users=0 --lim-tau=60 ; } &>> log/main/tau.log
echo "lim-tau=70 ..."
{ time bash run.sh --lim-users=0 --lim-tau=70 ; } &>> log/main/tau.log
echo "lim-tau=80 ..."
{ time bash run.sh --lim-users=0 --lim-tau=80 ; } &>> log/main/tau.log
echo "lim-tau=90 ..."
{ time bash run.sh --lim-users=0 --lim-tau=90 ; } &>> log/main/tau.log
echo "lim-tau=100 ..."
{ time bash run.sh --lim-users=0 --lim-tau=100 ; } &>> log/main/tau.log
echo "lim-tau=110 ..."
{ time bash run.sh --lim-users=0 --lim-tau=110 ; } &>> log/main/tau.log
echo "lim-tau=0 ..."
{ time bash run.sh --lim-users=0 --lim-tau=0 ; } &>> log/main/tau.log
