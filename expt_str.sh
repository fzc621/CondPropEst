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


for i in $(seq 0 5); do
  	echo "Run = ${i}"
  	sh expt.sh strength $i
done
