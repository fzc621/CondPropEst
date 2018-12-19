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

sweeps="0.01 0.05 0.1 0.5 1 5 10"

for i in $(seq 0 5); do
  for s in $sweeps; do
  	echo "#Sweep = ${s} Run = ${i}"
  	run "sh ./expt.sh sweep $s $i"
  done
done
