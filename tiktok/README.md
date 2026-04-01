# tiktok

The idea is to extract all the information that we can for the artists in our DB. For that, we first look for tiktok accounts in [Wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page) and in Musicbrainz and then we do the API calls.

- `get_data.sh` gets a JSONL with information about all entities inside Wikidata with a tiktok account associated.
    - `data/tiktok_musicbrainz_all.jsonl`
- `mb_tiktok_mapping.ipynb` gets the mapping for artist-tiktokusername stored in Musicbrainz.
    - `data/mb_tiktok_mapping.csv`
- `initialize_tiktok_attrs.py` initializes the necessary properties on Artist nodes in Neo4j (e.g., `tiktok_accounts`, `likes_count`, etc.) to prepare for data ingestion.
- `update_mb_ids.py` is in charge of updating our artist nodes with their MusicBrainz GIDs (only those with a tiktok account associated in Wikidata).
- `from_mb_to_artist.py` imports the TikTok usernames extracted from MusicBrainz (via `mb_tiktok_mapping.ipynb`) into Neo4j.
- `from_wd_to_artist.py` imports the TikTok usernames extracted from Wikidata (via `get_data.sh`) into Neo4j.
- `tiktok_requests.ipynb` is a notebook for testing and exploring the TikTok Research API.
- `call_tiktok.py` queries the TikTok Research API to fetch metrics (followers, likes, videos) and following lists for artists, updating the Neo4j database.
- `build_following_links.py` computes the Jaccard similarity index based on shared following lists and creates weighted `FOLLOWS` relationships between artists in Neo4j.
