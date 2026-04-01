#!/usr/bin/env python3


from neo4j import GraphDatabase, basic_auth, Driver
from dotenv import load_dotenv
import os
import json
import logging
import traceback
import tqdm
from itertools import batched
from typing import List, Dict, Any


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


def stream_wd_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """Stream and parse the JSONL file, producing cleaned rows.

    Each returned row has: {"gids": [str], "names_lc": [str], "usernames": [str]}
    - gids are the raw musicbrainz_ids (strings)
    - names_lc are lowercased names for case-insensitive matching
    - usernames are TikTok usernames (unique per row)
    """
    assert os.path.exists(jsonl_path), f"{jsonl_path} not found"

    rows: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                logging.warning("Skipping line due to parse error", exc_info=True)
                continue

            names = obj.get("names", []) or []
            gids = obj.get("musicbrainz_ids", []) or []
            tiktoks = obj.get("tiktoks", []) or []

            # Clean and normalize
            names_lc = sorted({str(n).strip().lower() for n in names if str(n).strip()})
            gids_clean = sorted({str(g).strip() for g in gids if str(g).strip()})
            usernames = sorted({str(u).strip() for u in tiktoks if str(u).strip()})

            if not usernames:
                continue

            if not names_lc and not gids_clean:
                # Nothing to match against
                continue

            rows.append({
                "gids": gids_clean,
                "names_lc": names_lc,
                "usernames": usernames,
            })

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

        jsonl_path = os.getenv(
            "WD_TIKTOK_JSONL", "data/tiktok_musicbrainz_all.jsonl"
        )
        rows = stream_wd_jsonl(jsonl_path)
        logging.info(f"Prepared {len(rows)} update rows from JSONL")

        batch_size = int(os.getenv("WD_BATCH_SIZE", "300"))

        with driver.session() as session:
            for batch in tqdm.tqdm(batched(rows, batch_size), desc="Processing WD batches"):
                updates_param = list(batch)
                # Use APOC to update in batches; match by known_gids OR known_names (case-insensitive)
                session.run(
                    """
                    CALL apoc.periodic.iterate(
                        'UNWIND $updates AS u RETURN u',
                        'MATCH (a:Artist)\nWHERE (size(u.gids) > 0 AND any(g IN a.known_gids WHERE g IN u.gids))\n   OR (size(u.gids) = 0 AND size(u.names_lc) > 0 AND any(n IN a.known_names_lc WHERE n IN u.names_lc))\nSET a.tiktok_accounts = apoc.coll.toSet(a.tiktok_accounts + u.usernames)',
                        {batchSize: 30, parallel: true, params: {updates: $updates}}
                    )
                    """,
                    updates=updates_param,
                )

            # Final deduplication pass across all artists having tiktok_accounts
            logging.info("Final deduplication pass across all artists having tiktok_accounts")
            session.run(
                """
                MATCH (a:Artist)
                WHERE a.tiktok_accounts IS NOT NULL AND size(a.tiktok_accounts) > 0
                SET a.tiktok_accounts = apoc.coll.toSet(a.tiktok_accounts)
                """
            )

        logging.info("TikTok accounts successfully updated and deduplicated in Neo4j")

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
