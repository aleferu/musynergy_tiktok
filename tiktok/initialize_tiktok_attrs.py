#!/usr/bin/env python3


from neo4j import GraphDatabase, basic_auth, Driver, ManagedTransaction
from dotenv import load_dotenv
import os
import logging
import traceback


def get_driver() -> Driver:
    """Create and return a Neo4j driver using environment variables.

    Expected env vars: NEO4J_HOST, NEO4J_PORT, NEO4J_USER, NEO4J_PASS
    """
    NEO4J_HOST = os.getenv("NEO4J_HOST")
    NEO4J_PORT = os.getenv("NEO4J_PORT")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASS = os.getenv("NEO4J_PASS")

    # .env validation (follow update_mb_ids.py style)
    assert NEO4J_HOST is not None and \
        NEO4J_PORT is not None and \
        NEO4J_USER is not None and \
        NEO4J_PASS is not None, \
        "INVALID .env"

    # db connection
    return GraphDatabase.driver(
        f"bolt://{NEO4J_HOST}:{NEO4J_PORT}",
        auth=basic_auth(NEO4J_USER, NEO4J_PASS)
    )


def initialize_tiktok_accounts(tx: ManagedTransaction):
    """Ensure every Artist node has a tiktok_accounts property as an empty list.

    This mirrors the initialization pattern used for known_gids in update_mb_ids.py,
    but targets the tiktok_accounts attribute on Artist nodes.
    """
    tx.run(
        """
        MATCH (a:Artist)
        WHERE a.tiktok_accounts IS NULL
        SET a.tiktok_accounts = [],
            a.tiktok_call = false,
            a.in_tiktok = false,
            a.likes_count = -1,
            a.video_count = -1,
            a.follower_count = -1,
            a.following_count = -1,
            a.following_list = []
        """
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    load_dotenv()
    driver = None

    try:
        driver = get_driver()
        logging.info("Access to Neo4j granted")

        logging.info("Initializing property tiktok_accounts if necessary")
        with driver.session() as session:
            session.execute_write(initialize_tiktok_accounts)

        logging.info("tiktok_accounts successfully initialized in Neo4j")

    except Exception:
        logging.error("An error occurred during initialization.")
        logging.error(traceback.format_exc())

    finally:
        if driver:
            driver.close()
            logging.info("Neo4j driver closed.")
        logging.info("Script finished (successfully or with error).")


if __name__ == "__main__":
    main()
