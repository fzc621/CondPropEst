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

n1="8 9 10 11 12 13 14 15 16 32 64"

for n in $n1; do
	echo "N1 = ${n}"
	run "sh ./expt.sh $n1"
done
