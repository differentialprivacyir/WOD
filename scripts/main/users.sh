#!/bin/bash

mkdir -p log/main

echo "lim-users=100 ..."
{ time bash run.sh --lim-users=100 ; } &>> log/main/users.log
echo "lim-users=1000 ..."
{ time bash run.sh --lim-users=1000  ; } &>> log/main/users.log
echo "lim-users=5000 ..."
{ time bash run.sh --lim-users=5000  ; } &>> log/main/users.log
echo "lim-users=10000 ..."
{ time bash run.sh --lim-users=10000 ; } &>> log/main/users.log
echo "lim-users=15000 ..."
{ time bash run.sh --lim-users=15000 ; } &>> log/main/users.log
echo "lim-users=20000 ..."
{ time bash run.sh --lim-users=20000 ; } &>> log/main/users.log
echo "lim-users=25000 ..."
{ time bash run.sh --lim-users=25000 ; } &>> log/main/users.log
echo "lim-users=30000 ..."
{ time bash run.sh --lim-users=30000 ; } &>> log/main/users.log
echo "lim-users=35000 ..."
{ time bash run.sh --lim-users=35000 ; } &>> log/main/users.log
echo "lim-users=40000 ..."
{ time bash run.sh --lim-users=40000 ; } &>> log/main/users.log
