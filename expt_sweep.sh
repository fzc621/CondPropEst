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

sweeps="0.0001 0.001 0.01 0.1 0.2 0.5 0.7 0.9 1 2 5 10 20"

for s in $sweeps; do
  for i in $(seq 0 5); do
  	echo "#Sweep = ${s} Run = ${i}"
  	run "sh ./expt.sh power sweep $s $i"
  done
done

# python -m src.eval_sweep -k 6 data/set1bin_power/expt/sweep

# python -m src.eval_rank -k 6 -m 10 data/set1bin_power/expt/res
