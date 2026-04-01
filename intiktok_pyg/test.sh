#!/usr/bin/env bash


set -xe


YEARS=(2023 2021 2019)
PERC_VALUES=(0 0.5 0.75 0.9)

for YEAR in "${YEARS[@]}"; do
    for PERC in "${PERC_VALUES[@]}"; do
        echo "Testing Year: $YEAR, Perc: $PERC"
        python3 intiktok_pyg/test.py --year "$YEAR" --perc "$PERC"
    done
done
