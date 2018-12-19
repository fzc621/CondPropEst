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

sweeps="1 2 5 10 20"

for i in $(seq 0 5); do
  for s in $sweeps; do
  	echo "#Sweep = ${s} Run = ${i}"
  	run "sh ./expt.sh learn $s $i"
  done
done
