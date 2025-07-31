#!/bin/bash

mkdir -p log/main_PM

echo "eps=0.5 ..."
{ time bash run.sh --lim-users=0 --eps=0.5 --alg=main_PM ; } &>> log/main_PM/eps.log
echo "eps=1 ..."
{ time bash run.sh --lim-users=0 --eps=1 --alg=main_PM ; } &>> log/main_PM/eps.log
echo "eps=1.5 ..."
{ time bash run.sh --lim-users=0 --eps=1.5 --alg=main_PM ; } &>> log/main_PM/eps.log
echo "eps=2 ..."
{ time bash run.sh --lim-users=0 --eps=2 --alg=main_PM ; } &>> log/main_PM/eps.log
echo "eps=2.5 ..."
{ time bash run.sh --lim-users=0 --eps=2.5 --alg=main_PM ; } &>> log/main_PM/eps.log
echo "eps=3 ..."
{ time bash run.sh --lim-users=0 --eps=3 --alg=main_PM ; } &>> log/main_PM/eps.log
echo "eps=3.5 ..."
{ time bash run.sh --lim-users=0 --eps=3.5 --alg=main_PM ; } &>> log/main_PM/eps.log
echo "eps=4 ..."
{ time bash run.sh --lim-users=0 --eps=4 --alg=main_PM ; } &>> log/main_PM/eps.log
echo "eps=5 ..."
{ time bash run.sh --lim-users=0 --eps=5 --alg=main_PM ; } &>> log/main_PM/eps.log
echo "eps=6 ..."
{ time bash run.sh --lim-users=0 --eps=6 --alg=main_PM ; } &>> log/main_PM/eps.log
echo "eps=8 ..."
{ time bash run.sh --lim-users=0 --eps=8 --alg=main_PM ; } &>> log/main_PM/eps.log
echo "eps=10 ..."
{ time bash run.sh --lim-users=0 --eps=10 --alg=main_PM ; } &>> log/main_PM/eps.log
