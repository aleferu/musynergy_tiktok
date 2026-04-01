#!/usr/bin/env python3


import csv
import argparse
import os.path as path
from neo4j import GraphDatabase, basic_auth
from dotenv import load_dotenv
import os
import tqdm


def get_most_famous_collab(tx, artist_a_id, artist_b_id, year):
    """
    Cypher query to find the track with the highest popularity where both artists
    have a WORKED_IN relationship.
    """
    query = """
    MATCH (a:Artist {main_id: $id_a})-[:WORKED_IN]->(t:Track)<-[:WORKED_IN]-(b:Artist {main_id: $id_b})
    WITH
        a, b, t,
        toInteger(t.year) AS intYear,
        toInteger(t.month) AS intMonth
    WHERE
        (
            (intYear = $year + 1) OR
            (intYear = $year + 2 AND intMonth IS NOT NULL AND intMonth < 11)
        )
        AND a.main_tag IS NOT NULL
        AND b.main_tag IS NOT NULL
    MATCH (tg:Tag)-[:TAGS]->(t)
    WITH a, b, t, intYear, tg
    WHERE tg.name IS NOT NULL
    RETURN
        a.known_names[0] AS artist_a_name,
        a.main_tag AS artist_a_main_tag,
        b.known_names[0] AS artist_b_name,
        b.main_tag AS artist_b_main_tag,
        t.name AS song_name,
        intYear AS year,
        t.popularity_scaled AS popularity_scaled,
        collect(DISTINCT tg.name) AS track_genres
    ORDER BY popularity_scaled DESC
    LIMIT 1
    """
    result = tx.run(query, id_a=artist_a_id, id_b=artist_b_id, year=year)
    return result.single()


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Enrich GNN predictions with Neo4j track data")
    parser.add_argument("--year", type=int, default=2021, help="Dataset year")
    parser.add_argument("--perc", type=float, default=0.5, help="Dataset percentage")

    args = parser.parse_args()

    month = 11
    input_csv = f"./intiktok_pyg/results_{args.year}_{month}_{args.perc}.csv"
    output_csv = f"./intiktok_pyg/results_{args.year}_{month}_{args.perc}_final.csv"

    if not path.exists(input_csv):
        print(f"Error: Input file {input_csv} not found.")
        return

    # .env read
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

    results_to_write = []

    print(f"Connecting to Neo4j and processing {input_csv}...")

    with driver.session() as session:
        with open(input_csv, mode="r") as f:
            reader = csv.DictReader(f)
            for row in tqdm.tqdm(reader):
                a_id = row["artist_a"]
                b_id = row["artist_b"]

                # Query Neo4j for the top collaboration
                record = session.execute_read(get_most_famous_collab, a_id, b_id, args.year)

                if record:
                    results_to_write.append({
                        "artist_a_id": a_id,
                        "artist_a_name": record["artist_a_name"],
                        "artist_a_main_tag": record["artist_a_main_tag"],
                        "artist_b_id": b_id,
                        "artist_b_name": record["artist_b_name"],
                        "artist_b_main_tag": record["artist_b_main_tag"],
                        "song_name": record["song_name"],
                        "year": record["year"],
                        "track_genres": record["track_genres"]
                    })

    driver.close()

    # Write the final enriched CSV
    headers = ["artist_a_id", "artist_a_name", "artist_a_main_tag", "artist_b_id", "artist_b_name", "artist_b_main_tag", "song_name", "year", "track_genres"]
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results_to_write)

    print(f"Successfully wrote {len(results_to_write)} rows to {output_csv}")


if __name__ == "__main__":
    main()
