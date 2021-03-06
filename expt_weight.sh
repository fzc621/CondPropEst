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

weights="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1"

for i in $(seq 0 5); do
  for s in $weights; do
  	echo "Context = ${s} Run = ${i}"
  	run "sh ./expt.sh weight $s $i"
  done
done
