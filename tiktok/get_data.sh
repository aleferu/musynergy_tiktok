#!/usr/bin/env bash

set -xe

# https://www.wikidata.org/wiki/Wikidata:Database_download

gzip -dc data/latest-all.json.gz \
  | npx wikibase-dump-filter --claim P7085 --simplify \
  > data/wikidata_tiktok.ndjson


jq -c '
  {
    names: ([.labels[]?] | unique),
    musicbrainz_ids: ([.claims.P434[]?] | unique),
    tiktoks: ([.claims.P7085[]?] | unique)
  }
' data/wikidata_tiktok.ndjson > data/tiktok_musicbrainz_all.jsonl
