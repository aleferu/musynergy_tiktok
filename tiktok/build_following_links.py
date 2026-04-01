#!/usr/bin/env python3


from dotenv import Any, load_dotenv
import os
from neo4j import GraphDatabase, basic_auth, Driver
import logging
from tqdm import tqdm


def get_driver() -> Driver:
    """Create and return a Neo4j driver using environment variables."""
    NEO4J_HOST = os.getenv("NEO4J_HOST")
    NEO4J_PORT = os.getenv("NEO4J_PORT")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASS = os.getenv("NEO4J_PASS")

    assert (
        NEO4J_HOST is not None
        and NEO4J_PORT is not None
        and NEO4J_USER is not None
        and NEO4J_PASS is not None
    ), "INVALID .env"

    return GraphDatabase.driver(
        f"bolt://{NEO4J_HOST}:{NEO4J_PORT}", auth=basic_auth(NEO4J_USER, NEO4J_PASS)
    )


def get_artist_info(driver: Driver) -> list[dict[str, Any]]:
    """[{"main_id": "1234", "tiktok_accounts": ["foo", "bar"], "following_list": ["baz", "asdf"]}]"""
    cypher = """
        MATCH (a:Artist)
        WHERE size(coalesce(a.tiktok_accounts, [])) > 0
        RETURN a.main_id AS main_id, a.tiktok_accounts AS tiktok_accounts, a.following_list AS following_list
    """
    with driver.session() as session:
        result = session.run(cypher)
        rows = [dict(r) for r in result]
    return rows


def get_tiktok_account_artist_mapping(artists: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """{ "tiktok_account": {"main_id": "1234", "tiktok_accounts": ["foo", "bar"], "following_list": ["baz", "asdf"]} }"""
    return {
        tiktok_account: artist
        for artist in artists for tiktok_account in artist["tiktok_accounts"]
    }


def get_jaccard_index_from_artists(a0: dict[str, Any], a1: dict[str, Any]):
    """Jaccard index of following lists"""
    a0_fl = set(a0["following_list"])
    a1_fl = set(a1["following_list"])
    intersection = a0_fl.intersection(a1_fl)
    union = a0_fl.union(a1_fl)
    return len(intersection) / len(union)


def update_neo4j_links(driver: Driver, updates: list[tuple[str, float, str]], batch_size: int = 5000):
    """Batch update Neo4j with similarity links."""
    query = """
        UNWIND $batch AS row
        MATCH (source:Artist {main_id: row[0]})
        MATCH (target:Artist {main_id: row[2]})
        MERGE (source)-[r:FOLLOWS]->(target)
        SET r.weight = row[1]
    """

    total = len(updates)
    logging.info("Starting batch update of %d links...", total)

    with driver.session() as session:
        # Loop through the list in chunks
        for i in range(0, total, batch_size):
            chunk = updates[i : i + batch_size]
            session.run(query, batch=chunk)
            logging.info("Progress: %d/%d links created", min(i + batch_size, total), total)

    logging.info("Successfully updated Neo4j!")


def main():
    logging.info("Getting Neo4j driver")
    driver = get_driver()

    logging.info("Collecting artists info")
    artists = get_artist_info(driver)
    logging.info("Got %d artists", len(artists))

    logging.info("Generating account->id mapping")
    tiktok_account_artist_mapping = get_tiktok_account_artist_mapping(artists)
    logging.info("Got %d mappings", len(tiktok_account_artist_mapping))

    updates: list[tuple[str, float, str]] = list()

    logging.info("Iterating through the artists and computing the updates...")
    for a0 in tqdm(artists):
        a0_id = a0["main_id"]
        a0_fl = a0["following_list"]
        for a1_tt in a0_fl:
            a1 = tiktok_account_artist_mapping.get(a1_tt)
            if a1 and a1 is not a0:
                jaccard = get_jaccard_index_from_artists(a0, a1)
                updates.append((a0_id, jaccard, a1["main_id"]))
    logging.info("Got %d updates", len(updates))

    if len(updates) > 0:
        update_neo4j_links(driver, updates)

    logging.info("Closing Neo4j driver...")
    driver.close()
    logging.info("Closed!")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    load_dotenv()
    main()
