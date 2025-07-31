#!/bin/bash

mkdir -p log/main_Duchi

echo "lim-users=100 ..."
{ time bash run.sh --lim-users=100 --alg=main_Duchi ; } &>> log/main_Duchi/users.log
echo "lim-users=1000 ..."
{ time bash run.sh --lim-users=1000 --alg=main_Duchi ; } &>> log/main_Duchi/users.log
echo "lim-users=5000 ..."
{ time bash run.sh --lim-users=5000 --alg=main_Duchi ; } &>> log/main_Duchi/users.log
echo "lim-users=10000 ..."
{ time bash run.sh --lim-users=10000 --alg=main_Duchi ; } &>> log/main_Duchi/users.log
echo "lim-users=15000 ..."
{ time bash run.sh --lim-users=15000 --alg=main_Duchi ; } &>> log/main_Duchi/users.log
echo "lim-users=20000 ..."
{ time bash run.sh --lim-users=20000 --alg=main_Duchi ; } &>> log/main_Duchi/users.log
echo "lim-users=25000 ..."
{ time bash run.sh --lim-users=25000 --alg=main_Duchi ; } &>> log/main_Duchi/users.log
echo "lim-users=30000 ..."
{ time bash run.sh --lim-users=30000 --alg=main_Duchi ; } &>> log/main_Duchi/users.log
echo "lim-users=30000 ..."
{ time bash run.sh --lim-users=35000 --alg=main_Duchi ; } &>> log/main_Duchi/users.log
echo "lim-users=40000 ..."
{ time bash run.sh --lim-users=40000 --alg=main_Duchi ; } &>> log/main_Duchi/users.log
