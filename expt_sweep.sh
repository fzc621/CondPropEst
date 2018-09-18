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

# sweeps="5"
sweeps="1 2 5 10 20"

for s in $sweeps; do
  for i in $(seq 0 5); do
  	echo "#Sweep = ${s} Run = ${i}"
  	run "sh ./expt.sh power sweep $s $i"
  done
done
