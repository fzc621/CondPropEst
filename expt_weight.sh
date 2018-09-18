#!/usr/bin/env bash

set -e

run()
{
  if [[ "$OSTYPE" == "linux-gnu" ]]; then
    ~/submit_job.pl "$1"
  else
    $1
  fi
}

weights="0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.095"

for s in $weights; do
  for i in $(seq 0 5); do
  	echo "#Sweep = ${s} Run = ${i}"
  	run "sh ./expt.sh power weight $s $i"
  done
done
