#!/usr/bin/env python3


from neo4j import GraphDatabase, basic_auth, Driver
from dotenv import load_dotenv
import os
import logging
import traceback
from typing import Dict, List, Any, Optional
import tqdm
import time

import requests


class RateLimitError(Exception):
    """Raised when TikTok API returns HTTP 429 (Too Many Requests)."""
    pass


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


def fetch_artists_to_call(driver: Driver, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetch artists with TikTok accounts pending call.

    Returns list of dicts: {"id": a.main_id, "accounts": [str, ...]}
    """
    cypher = (
        "MATCH (a:Artist)\n"
        "WHERE size(coalesce(a.tiktok_accounts, [])) > 0 "
        "AND (a.tiktok_call = false OR a.tiktok_call IS NULL)\n"
        "RETURN a.main_id AS id, a.tiktok_accounts AS accounts"
    )
    if limit is not None and limit > 0:
        cypher += " LIMIT $limit"

    with driver.session() as session:
        result = session.run(cypher, limit=limit) if limit else session.run(cypher)
        rows = [dict(r) for r in result]
    return rows


def merge_artist_metrics(per_account: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge per-account metrics for an artist.

    Numeric fields are summed; following_list is the union of all usernames.
    """
    likes = 0
    videos = 0
    followers = 0
    following_cnt = 0
    following_set = set()

    for m in per_account:
        likes += int(m.get("likes_count", 0) or 0)
        videos += int(m.get("video_count", 0) or 0)
        followers += int(m.get("follower_count", 0) or 0)
        following_cnt += int(m.get("following_count", 0) or 0)
        fl = m.get("following_list", []) or []
        if isinstance(fl, list):
            following_set.update(str(x) for x in fl if str(x))

    return {
        "likes_count": likes,
        "video_count": videos,
        "follower_count": followers,
        "following_count": following_cnt,
        "following_list": list(following_set),
    }


def get_access_token(client_key: str, client_secret: str) -> Optional[str]:
    """Get TikTok Research API access token using client credentials flow."""
    url = "https://open.tiktokapis.com/v2/oauth/token/"
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Cache-Control": "no-cache"}
    data = {
        "client_key": client_key,
        "client_secret": client_secret,
        "grant_type": "client_credentials",
        "scope": "research.data.basic",
    }
    try:
        resp = requests.post(url, headers=headers, data=data, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        tok = payload.get("access_token")
        if not tok:
            logging.error("Token response missing access_token: %s", payload)
            return None
        return tok
    except Exception:
        logging.error("Failed to obtain TikTok access token", exc_info=True)
        return None


def get_tiktok_user_info(username: str, auth_token: str) -> Optional[Dict[str, Any]]:
    """Fetch user info for a username; return None on error.

    Logs whenever data is missing or the user is not verified so we know why we skip.
    """
    url = "https://open.tiktokapis.com/v2/research/user/info/"
    headers = {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}
    params = {"fields": "is_verified,follower_count,following_count,likes_count,video_count"}
    body = {"username": username}
    try:
        resp = requests.post(url, headers=headers, params=params, json=body, timeout=15)
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            # Include response text for easier debugging
            status = getattr(http_err.response, "status_code", None)
            text = getattr(http_err.response, "text", "<no text>")
            logging.warning("HTTP error in user info for @%s (status=%s): %s", username, status, text)
            if status == 429:
                logging.error("HTTP 429 received in user info for @%s. Stopping the run due to rate limiting.", username)
                raise RateLimitError("TikTok API rate limit (429) in user info")
            raise
        payload = resp.json()
        data = payload.get("data", {})
        if not data:
            logging.info("No data returned for @%s; payload keys: %s", username, list(payload.keys()))
            return None
        logging.info(
            "Got user info for @%s: followers=%s following=%s likes=%s videos=%s",
            username,
            data.get("follower_count"),
            data.get("following_count"),
            data.get("likes_count"),
            data.get("video_count"),
        )
        return data
    except RateLimitError:
        # Bubble up to stop the entire process
        raise
    except Exception:
        logging.warning("Failed to get user info for @%s", username, exc_info=True)
        return None


def get_all_following_list(username: str, auth_token: str, page_size: int = 100) -> Optional[List[str]]:
    """Paginate over following list and return usernames. Returns None on error.

    Note: No artificial max_total cap; we paginate until the API says no more.
    """
    url = "https://open.tiktokapis.com/v2/research/user/following/"
    headers = {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}
    params = {"fields": "display_name,bio_description,avatar_url,is_verified,follower_count,following_count,likes_count,video_count"}
    all_usernames: List[str] = []
    cursor: Optional[str] = None
    logging.info("Starting pagination to fetch following for @%s with page_size=%d", username, page_size)
    try:
        while True:
            body: Dict[str, Any] = {"username": username, "max_count": page_size}
            if cursor:
                body["cursor"] = cursor
            try:
                resp = requests.post(url, headers=headers, params=params, json=body, timeout=20)
                resp.raise_for_status()
            except requests.exceptions.HTTPError as http_err:
                status = getattr(http_err.response, "status_code", None)
                text = getattr(http_err.response, "text", "<no text>")
                logging.warning("HTTP error while fetching following for @%s (status=%s): %s", username, status, text)
                if status == 429:
                    logging.error("HTTP 429 received while fetching following for @%s. Stopping the run due to rate limiting.", username)
                    raise RateLimitError("TikTok API rate limit (429) in following list")
                raise
            payload = resp.json()
            data_block = payload.get("data", {})
            users = data_block.get("user_following", [])
            fetched_now = [u.get("username") for u in users if isinstance(u, dict) and u.get("username")]
            all_usernames.extend(fetched_now)
            has_more = data_block.get("has_more", False)
            cursor = data_block.get("cursor")
            logging.info("Fetched %d following in this page for @%s; total so far: %d; has_more=%s", len(fetched_now), username, len(all_usernames), has_more)
            if not has_more or not cursor or cursor == -1:
                break
        logging.info("Finished fetching following for @%s. Total collected: %d", username, len(all_usernames))
        return all_usernames
    except RateLimitError:
        # Bubble up to stop the entire run
        raise
    except Exception:
        logging.warning("Failed to get following list for @%s", username, exc_info=True)
        return None


def fetch_tiktok_account_metrics(auth_token: str, username: str, page_size: int) -> Optional[Dict[str, Any]]:
    """Fetch metrics for a single TikTok username using TikTok Research API (REST)."""
    info = get_tiktok_user_info(username, auth_token)
    if not info:
        logging.info("Skipping @%s: no user info returned (missing data or not verified)", username)
        return None
    likes = int(info.get("likes_count", 0) or 0)
    videos = int(info.get("video_count", 0) or 0)
    followers = int(info.get("follower_count", 0) or 0)
    following_cnt = int(info.get("following_count", 0) or 0)
    following_list = get_all_following_list(username, auth_token, page_size=page_size) or []
    logging.info("Fetched metrics for @%s: followers=%d, following=%d, likes=%d, videos=%d", username, followers, following_cnt, likes, videos)
    logging.info("Following list size for @%s: %d", username, len(following_list))
    return {
        "likes_count": likes,
        "video_count": videos,
        "follower_count": followers,
        "following_count": following_cnt,
        "following_list": following_list,
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    load_dotenv()
    driver: Optional[Driver] = None

    try:
        driver = get_driver()
        logging.info("Access to Neo4j granted")

        # Environment config
        client_key = os.getenv("TIKTOK_CLIENT_KEY")
        client_secret = os.getenv("TIKTOK_CLIENT_SECRET")
        assert client_key and client_secret, "Missing TIKTOK_CLIENT_KEY / TIKTOK_CLIENT_SECRET in .env"

        limit = int(os.getenv("TIKTOK_CALL_LIMIT", "0")) or None
        page_size = int(os.getenv("TIKTOK_FOLLOWING_PAGE_SIZE", "100"))

        # Obtain access token
        access_token = get_access_token(client_key, client_secret)
        if not access_token:
            logging.error("Could not obtain access token. Aborting.")
            return

        artists = fetch_artists_to_call(driver, limit=limit)
        logging.info("Found %d artists pending TikTok call", len(artists))
        if not artists:
            logging.info("Nothing to do.")
            return

        # Enforce 1h50m time cap due to TikTok token expiry window
        start_time = time.time()
        deadline = start_time + 110 * 60

        try:
            with driver.session() as session:
                for artist in tqdm.tqdm(artists, desc="Fetching TikTok metrics"):
                    if time.time() >= deadline:
                        logging.info("Time limit reached (1h50m). Stopping further processing.")
                        break

                    aid = artist.get("id")
                    accounts = artist.get("accounts") or []
                    if not aid or not isinstance(accounts, list) or not accounts:
                        continue

                    per_account_metrics: List[Dict[str, Any]] = []
                    for username in accounts:
                        uname = str(username).lstrip("@").strip()
                        if not uname:
                            logging.info("Skipping empty/invalid username in accounts for artist %s", aid)
                            continue
                        metrics = fetch_tiktok_account_metrics(access_token, uname, page_size)
                        if metrics is not None:
                            per_account_metrics.append(metrics)

                    if not per_account_metrics:
                        logging.info(
                            "Skipping artist %s: no metrics could be fetched for any account.", aid
                        )
                        try:
                            session.run(
                                """
                                MATCH (a:Artist {main_id: $id})
                                SET a.tiktok_call = true
                                """,
                                id=aid,
                            )
                            logging.info("Marked artist %s as tiktok_call=true due to skip.", aid)
                        except Exception:
                            logging.warning("Failed to mark artist %s as tiktok_call=true", aid, exc_info=True)
                        continue

                    merged = merge_artist_metrics(per_account_metrics)

                    try:
                        session.run(
                            """
                            MATCH (a:Artist {main_id: $id})
                            SET a.likes_count = coalesce($likes_count, 0),
                                a.video_count = coalesce($video_count, 0),
                                a.follower_count = coalesce($follower_count, 0),
                                a.following_count = coalesce($following_count, 0),
                                a.following_list = coalesce($following_list, []),
                                a.tiktok_call = true,
                                a.in_tiktok = true
                            """,
                            id=aid,
                            likes_count=merged.get("likes_count", 0),
                            video_count=merged.get("video_count", 0),
                            follower_count=merged.get("follower_count", 0),
                            following_count=merged.get("following_count", 0),
                            following_list=merged.get("following_list", []),
                        )
                        logging.info(
                            "Updated artist %s: likes=%d videos=%d followers=%d following=%d following_list=%d",
                            aid,
                            merged.get("likes_count", 0),
                            merged.get("video_count", 0),
                            merged.get("follower_count", 0),
                            merged.get("following_count", 0),
                            len(merged.get("following_list", [])),
                        )
                    except Exception:
                        logging.warning("Failed to update artist %s in Neo4j", aid, exc_info=True)
        except RateLimitError:
            logging.error("Rate limit encountered (HTTP 429). Stopping all processing immediately.")
            raise

        logging.info("TikTok metrics fetching and updates finished.")

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
