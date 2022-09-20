#!/bin/bash

python -m cProfile -o ./results/stat.prof -s time -s calls profiler_example.py
gprof2dot ./results/stat.prof -f pstats > ./results/stats.dot
dot -o ./results/stat.png -Tpng ./results/stats.dot
