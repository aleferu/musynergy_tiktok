#!/usr/bin/env bash


set -xe


YEARS=(2023 2021 2019)
PERC_VALUES=(0 0.5 0.75 0.9)

for YEAR in "${YEARS[@]}"; do
    for PERC in "${PERC_VALUES[@]}"; do
        python3 tiktok_pyg/train.py --year "$YEAR" --perc "$PERC" --original
    done
done
