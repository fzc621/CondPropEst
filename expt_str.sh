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


for i in $(seq 0 9); do
  	echo "Run = ${i}"
  	run "sh expt.sh strength $i"
done
