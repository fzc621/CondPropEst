#!/bin/sh

# python -m src.eval_rank -k 6 -m 10 data/set1bin/expt/rank
# #
# python -m src.eval_weight -k 6 data/set1bin/expt/weight
#
# python -m src.eval_sweep -k 6 data/set1bin/expt/sweep
#
# python -m src.eval_str -k 20 data/set1bin/expt/strength

python -m src.eval_learning -k 6 data/set1bin/expt/learn
