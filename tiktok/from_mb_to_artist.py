#!/usr/bin/env python3


from neo4j import GraphDatabase, basic_auth, Driver
from dotenv import load_dotenv
import pandas as pd
import os
import logging
import traceback
import tqdm
from itertools import batched


def get_driver() -> Driver:
    """Create and return a Neo4j driver using environment variables.

    Expected env vars: NEO4J_HOST, NEO4J_PORT, NEO4J_USER, NEO4J_PASS
    """
    NEO4J_HOST = os.getenv("NEO4J_HOST")
    NEO4J_PORT = os.getenv("NEO4J_PORT")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASS = os.getenv("NEO4J_PASS")

    # .env validation (follow update_mb_ids.py style)
    assert (
        NEO4J_HOST is not None
        and NEO4J_PORT is not None
        and NEO4J_USER is not None
        and NEO4J_PASS is not None
    ), "INVALID .env"

    # db connection
    return GraphDatabase.driver(
        f"bolt://{NEO4J_HOST}:{NEO4J_PORT}", auth=basic_auth(NEO4J_USER, NEO4J_PASS)
    )


def load_mapping(csv_path: str) -> list[dict]:
    """Load the mb→tiktok mapping and aggregate usernames per artist_id.

    Returns a list of dicts: {"id": str(artist_id), "usernames": [str, ...]}
    """
    assert os.path.exists(csv_path), f"{csv_path} not found"
    df = pd.read_csv(csv_path)

    # Basic validation and cleaning
    assert "artist_id" in df.columns and "tiktok_username" in df.columns, (
        "CSV must contain 'artist_id' and 'tiktok_username' columns"
    )

    df = df.dropna(subset=["artist_id", "tiktok_username"]).copy()
    # Ensure artist_id is string to match known_ids (which are strings)
    df["artist_id"] = df["artist_id"].astype(str)
    df["tiktok_username"] = df["tiktok_username"].astype(str).str.strip()
    # Remove empty usernames after strip
    df = df[df["tiktok_username"] != ""]

    # Aggregate usernames per artist_id and deduplicate
    grouped = (
        df.groupby("artist_id")["tiktok_username"]
        .apply(lambda s: sorted(set(s.tolist())))
        .reset_index()
        .rename(columns={"artist_id": "id", "tiktok_username": "usernames"})
    )

    rows = grouped.to_dict("records")
    return rows


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    load_dotenv()
    driver: Driver | None = None

    try:
        driver = get_driver()
        logging.info("Access to Neo4j granted")

        csv_path = os.getenv(
            "MB_TIKTOK_MAPPING_CSV", "data/mb_tiktok_mapping.csv"
        )
        updates = load_mapping(csv_path)
        logging.info(f"Prepared {len(updates)} artist updates from mapping CSV")

        batch_size = 300
        with driver.session() as session:
            for batch in tqdm.tqdm(
                batched(updates, batch_size), desc="Processing batches"
            ):
                updates_param = list(batch)
                # Use APOC to update in batches
                session.run(
                    """
                    CALL apoc.periodic.iterate(
                        'UNWIND $updates AS u RETURN u',
                        'MATCH (a:Artist)\nWHERE u.id IN a.known_ids\nSET a.tiktok_accounts = apoc.coll.toSet(a.tiktok_accounts + u.usernames)',
                        {batchSize: 50, parallel: true, params: {updates: $updates}}
                    )
                    """,
                    updates=updates_param,
                )

        logging.info("TikTok accounts successfully updated in Neo4j")

    except Exception:
        logging.error("An error occurred during processing.")
        logging.error(traceback.format_exc())

    finally:
        if driver:
            driver.close()
            logging.info("Neo4j driver closed.")
        logging.info("Script finished (successfully or with error).")


if __name__ == "__main__":
    main()
