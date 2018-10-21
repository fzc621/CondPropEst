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

weights="0.02 0.04 0.06 0.08 0.1 0.12 0.14"

for s in $weights; do
  for i in $(seq 0 5); do
  	echo "#Sweep = ${s} Run = ${i}"
  	run "sh ./expt.sh weight $s $i"
  done
done

# python -m src.eval_weight -k 6 data/set1bin_power/expt/weight
