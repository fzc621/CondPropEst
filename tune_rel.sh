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

paras="8 16 32 64 128 256"

for p in $paras; do
	echo "#N2 = ${p}"
	run "sh ./expt.sh $p"
done
