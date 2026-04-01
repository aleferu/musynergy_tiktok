#!/usr/bin/env python3


from neo4j import GraphDatabase, basic_auth, Driver
from dotenv import load_dotenv
import os
import logging
from itertools import combinations_with_replacement
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main(driver: Driver, decades: list[int], filename: str) -> None:
    # We use the decades directly as our labels
    labels = [f"{d}-{d+9}" for d in decades]

    heatmap_data = pd.DataFrame(
        index=labels,  # type: ignore
        columns=labels,  # type: ignore
        dtype=int
    ).fillna(0).astype(int)

    # Querying Neo4j
    with driver.session() as session:
        for d0, d1 in combinations_with_replacement(decades, 2):
            logging.info("Querying for decade %d and %d...", d0, d1)
            # Query filters by the decade range for Artist.begin_date
            query = f"""
                MATCH (n:Artist)-[r:FOLLOWS]->(m:Artist)
                WHERE n.in_tiktok = true AND m.in_tiktok = true
                AND n.begin_date >= {d0} AND n.begin_date <= {d0 + 9}
                AND m.begin_date >= {d1} AND m.begin_date <= {d1 + 9}
                AND n < m
                WITH COUNT(r) AS c
                RETURN toInteger(c) AS c;
            """

            result = session.run(query)  # type: ignore
            count = result.data()[0]["c"]

            logging.info("Got: %d", count)

            label0 = f"{d0}-{d0+9}"
            label1 = f"{d1}-{d1+9}"

            heatmap_data.loc[label0, label1] += count  # type: ignore
            heatmap_data.loc[label1, label0] += count  # type: ignore

    # Normalization by row
    row_sums = heatmap_data.sum(axis=1)
    normalized_heatmap_data = heatmap_data.div(row_sums, axis=0).fillna(0)

    # Annotations with percentage and raw count
    annot_labels = (
        normalized_heatmap_data.applymap(lambda x: f"{x:.1%}") +  # type: ignore
        "\n(" +
        heatmap_data.astype(str) + ")"
    )

    # Drawing
    logging.info("Generating png...")

    fig_size = 12

    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        normalized_heatmap_data,
        annot=annot_labels,
        cmap="coolwarm",
        xticklabels=True,
        yticklabels=True,
        cbar=True,
        square=True,
        fmt="",
        linewidths=.5,
        linecolor='black',
        cbar_kws={"shrink": .8},
        # annot_kws={"size": 9},
        vmin=0,
        vmax=1
    )
    plt.tight_layout()
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.savefig(f"img/{filename}_tiktok.png", dpi=300)


if __name__ == '__main__':
    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # .env read
    load_dotenv()
    DB_HOST = os.getenv("NEO4J_HOST")
    DB_PORT = os.getenv("NEO4J_PORT")
    DB_USER = os.getenv("NEO4J_USER")
    DB_PASS = os.getenv("NEO4J_PASS")

    # .env validation
    assert DB_HOST is not None and \
        DB_PORT is not None and \
        DB_USER is not None and \
        DB_PASS is not None, \
        "INVALID .env"

    # db connection
    driver = GraphDatabase.driver(f"bolt://{DB_HOST}:{DB_PORT}", auth=basic_auth(DB_USER, DB_PASS))

    # Decades from 1940 to 2020
    decades_list = [1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020]

    main(driver, decades_list, "decades_follows_tags")

    driver.close()

    logging.info("DONE!")
