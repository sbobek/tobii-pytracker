#!/bin/bash

for i in {0..17}
do
    python3 plot_gaze_data.py --input_csv="../data/data${i}.csv"
done
