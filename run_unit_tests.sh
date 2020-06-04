#!/usr/bin/env bash

for family in 0 1
 do
    for dataset in 0
    do
        for model in 0
        do
            for xl in 0
            do
                python -u code/run.py $attack -f $family -d $dataset -m $model --epochs 2 -v 
            done
        done
    done
done
