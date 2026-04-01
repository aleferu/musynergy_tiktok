#!/usr/bin/env python3


from neo4j import GraphDatabase, basic_auth, Driver, ManagedTransaction
from sqlalchemy import create_engine, Engine, text
from dotenv import load_dotenv
import pandas as pd
import os
import logging
import json
import tqdm
from itertools import batched
import traceback
import uuid


def is_valid_uuid(u):
    try:
        uuid.UUID(str(u))
        return True
    except ValueError:
        return False


def get_engine() -> Engine:
    DB_NAME = os.getenv("DB_NAME")
    DB_HOST = os.getenv("DB_HOST")
    DB_USER = os.getenv("DB_USER")
    DB_PASS = os.getenv("DB_PASS")
    DB_PORT = os.getenv("DB_PORT")

    # .env validation
    assert DB_NAME is not None and \
        DB_HOST is not None and \
        DB_PORT is not None and \
        DB_USER is not None and \
        DB_PASS is not None, \
        "INVALID .env"

    engine_url = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(engine_url, pool_size=10, max_overflow=0)
    return engine


def get_driver() -> Driver:
    NEO4J_HOST = os.getenv("NEO4J_HOST")
    NEO4J_PORT = os.getenv("NEO4J_PORT")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASS = os.getenv("NEO4J_PASS")

    # .env validation
    assert NEO4J_HOST is not None and \
        NEO4J_PORT is not None and \
        NEO4J_USER is not None and \
        NEO4J_PASS is not None, \
        "INVALID .env"

    # db connection
    return GraphDatabase.driver(f"bolt://{NEO4J_HOST}:{NEO4J_PORT}", auth=basic_auth(NEO4J_USER, NEO4J_PASS))


def initialize_known_gids(tx: ManagedTransaction):
    """Ensure every Artist node has a known_gids property as an empty list."""
    tx.run("""
        MATCH (a:Artist)
        WHERE a.known_gids IS NULL
        SET a.known_gids = []
    """)


def add_gid_to_artist(tx: ManagedTransaction, artist_id: str, gid: str):
    """Append gid to known_gids for the artist if not already present."""
    tx.run("""
        MATCH (a:Artist)
        WHERE $id IN a.known_ids
        SET a.known_gids = CASE
            WHEN NOT $gid IN a.known_gids THEN a.known_gids + [$gid]
            ELSE a.known_gids
        END
    """, id=artist_id, gid=gid)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    load_dotenv()
    engine = None
    driver = None

    try:
        engine = get_engine()
        driver = get_driver()
        logging.info("Access to DBs granted")

        jsonl_path = "data/tiktok_musicbrainz_all.jsonl"
        assert os.path.exists(jsonl_path), f"{jsonl_path} not found"

        # Step 1: Load all GIDs from the JSONL file
        gids = set()
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    possible_gids = obj.get("musicbrainz_ids", [])
                    valid_gids = [gid for gid in possible_gids if is_valid_uuid(gid)]
                    gids.update(valid_gids)
                except Exception as e:
                    logging.warning(f"Skipping line due to parse error: {e}")

        logging.info(f"Loaded {len(gids)} unique GIDs from JSONL")

        # Step 2: Get mapping from gid → id from PostgreSQL
        query = text("""
            SELECT id, gid 
            FROM artist 
            WHERE gid IN (SELECT UNNEST(CAST(:gids AS uuid[])))
        """)
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn, params={"gids": list(gids)})


        logging.info(f"Found {len(df)} artist matches in PostgreSQL")

        # Step 3: Initialize empty list property for all artists in Neo4j
        logging.info("Initializing property known_gids if necessary")
        with driver.session() as session:
            session.execute_write(initialize_known_gids)

        # Step 4: Add GIDs to Neo4j nodes
        rows = df.to_dict("records")
        batch_size = 150

        with driver.session() as session:
            for batch in tqdm.tqdm(batched(rows, batch_size), desc="Processing batches"):
                # Prepare batch param
                updates_param = [{"id": str(r["id"]), "gid": str(r["gid"])} for r in batch]

                # APOC periodic iterate for Neo4j
                session.run("""
                    CALL apoc.periodic.iterate(
                        'UNWIND $updates AS u RETURN u',
                        'MATCH (a:Artist)
                         WHERE u.id IN a.known_ids
                         SET a.known_gids = CASE
                             WHEN NOT u.gid IN a.known_gids THEN a.known_gids + [u.gid]
                             ELSE a.known_gids
                         END',
                        {batchSize:30, parallel:true, params:{updates:$updates}}
                    )
                """, updates=updates_param)

        logging.info("GIDs successfully updated in Neo4j")

    except Exception as e:
        logging.error("An error occurred during processing.")
        logging.error(traceback.format_exc())

    finally:
        # Always close resources
        if driver:
            driver.close()
            logging.info("Neo4j driver closed.")
        if engine:
            engine.dispose()
            logging.info("PostgreSQL engine disposed.")
        logging.info("Script finished (successfully or with error).")

if __name__ == "__main__":
    main()
