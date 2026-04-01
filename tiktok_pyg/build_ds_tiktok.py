#!/usr/bin/env python3


from neo4j import Driver, GraphDatabase, basic_auth
from dotenv import load_dotenv
import os
import torch
import logging
import multiprocessing
import pickle


def get_x_count(x: str, driver: Driver) -> int:
    with driver.session() as session:
        query = f"MATCH {x} return COUNT(*) AS c;"
        return session.run(query).data()[0]["c"]  # type: ignore


def get_x_map(filepath: str) -> dict:
    with open(filepath, "rb") as in_file:
        loaded_dict = pickle.load(in_file)
    return loaded_dict


def get_artist_map() -> dict:
    logging.info("Loading artist dict...")
    return get_x_map("./tiktok_pyg/ds/artist_map.pkl")


def get_track_map() -> dict:
    logging.info("Loading track dict...")
    return get_x_map("./tiktok_pyg/ds/track_map.pkl")


def get_tag_map() -> dict:
    logging.info("Loading tag dict...")
    return get_x_map("./tiktok_pyg/ds/tag_map.pkl")


def build_artist_map():
    logging.info("Artist mapping...")
    driver = GraphDatabase.driver(f"bolt://{DB_HOST}:{DB_PORT}", auth=basic_auth(DB_USER, DB_PASS))  # type: ignore
    with driver.session() as session:
        query = "MATCH (n:Artist {tiktok_call: true}) RETURN n.main_id AS main_id;"
        q_result = session.run(query)
        result_map = {
            record["main_id"]: i
            for i, record in enumerate(q_result)
        }
        logging.info("Got %d records for artists", len(result_map))
        with open("./tiktok_pyg/ds/artist_map.pkl", "wb") as out_file:
            pickle.dump(result_map, out_file)
    driver.close()
    logging.info("Artist mapping done")


def build_track_map():
    logging.info("Track mapping...")
    driver = GraphDatabase.driver(f"bolt://{DB_HOST}:{DB_PORT}", auth=basic_auth(DB_USER, DB_PASS))  # type: ignore
    result_map = dict()
    with driver.session() as session:
        query = "MATCH (t:Track)<-[:WORKED_IN]-(a:Artist {tiktok_call: true}) RETURN DISTINCT t.id AS id;"
        q_result = session.run(query)
        result_map = {
            record["id"]: i
            for i, record in enumerate(q_result)
        }
        logging.info("Got %d records for tracks", len(result_map))
        with open("./tiktok_pyg/ds/track_map.pkl", "wb") as out_file:
            pickle.dump(result_map, out_file)
    driver.close()
    logging.info("Track mapping done")


def build_tag_map():
    logging.info("Tag mapping...")
    driver = GraphDatabase.driver(f"bolt://{DB_HOST}:{DB_PORT}", auth=basic_auth(DB_USER, DB_PASS))  # type: ignore
    result_map = dict()
    with driver.session() as session:
        query = """
            MATCH (a:Artist {tiktok_call: true})<-[:TAGS]-(t:Tag)
            RETURN t.id AS id
            UNION
            MATCH (:Artist {tiktok_call: true})-[:WORKED_IN]->(:Track)<-[:TAGS]-(t:Tag)
            RETURN t.id AS id
        """
        q_result = session.run(query)
        result_map = {
            record["id"]: i
            for i, record in enumerate(q_result)
        }
        logging.info("Got %d records for tags", len(result_map))
        with open("./tiktok_pyg/ds/tag_map.pkl", "wb") as out_file:
            pickle.dump(result_map, out_file)
    driver.close()
    logging.info("Tag mapping done")


def build_artist_tensor():
    artist_map = get_artist_map()
    logging.info("Building artist tensor...")
    driver = GraphDatabase.driver(f"bolt://{DB_HOST}:{DB_PORT}", auth=basic_auth(DB_USER, DB_PASS))  # type: ignore
    result_tensor = torch.empty((len(artist_map), 21 + 1), dtype=torch.float32)
    with driver.session() as session:
        query = """
            MATCH (n:Artist {tiktok_call: true})
            RETURN
                n.main_id as main_id,
                CASE WHEN n.begin_date IS NULL THEN 0 ELSE 1 END AS has_begin_date,
                COALESCE(n.begin_date, 0) AS begin_date,
                CASE WHEN n.end_date IS NULL THEN 0 else 1 END AS has_end_date,
                COALESCE(n.end_date, 0) AS end_date,
                n.ended AS ended,
                n.gender_1 AS gender_1,
                n.gender_2 AS gender_2,
                n.gender_3 AS gender_3,
                n.gender_4 AS gender_4,
                n.gender_5 AS gender_5,
                n.popularity_scaled AS popularity_scaled,
                n.type_1 AS type_1,
                n.type_2 AS type_2,
                n.type_3 AS type_3,
                n.type_4 AS type_4,
                n.type_5 AS type_5,
                n.type_6 AS type_6,
                n.follower_count AS follower_count,
                n.following_count AS following_count,
                n.likes_count AS likes_count,
                n.video_count AS video_count
            ;
        """
        q_result = session.run(query)
        for record in q_result:
            artist_idx = artist_map[record["main_id"]]
            result_tensor[artist_idx, 0]  = record["has_begin_date"]
            result_tensor[artist_idx, 1]  = record["begin_date"]
            result_tensor[artist_idx, 2]  = record["has_end_date"]
            result_tensor[artist_idx, 3]  = record["end_date"]
            result_tensor[artist_idx, 4]  = record["ended"]
            result_tensor[artist_idx, 5]  = record["gender_1"]
            result_tensor[artist_idx, 6]  = record["gender_2"]
            result_tensor[artist_idx, 7]  = record["gender_3"]
            result_tensor[artist_idx, 8]  = record["gender_4"]
            result_tensor[artist_idx, 9]  = record["gender_5"]
            result_tensor[artist_idx, 10] = record["popularity_scaled"]
            result_tensor[artist_idx, 11] = record["type_1"]
            result_tensor[artist_idx, 12] = record["type_2"]
            result_tensor[artist_idx, 13] = record["type_3"]
            result_tensor[artist_idx, 14] = record["type_4"]
            result_tensor[artist_idx, 15] = record["type_5"]
            result_tensor[artist_idx, 16] = record["type_6"]
            result_tensor[artist_idx, 17] = record["follower_count"]
            result_tensor[artist_idx, 18] = record["following_count"]
            result_tensor[artist_idx, 19] = record["likes_count"]
            result_tensor[artist_idx, 20] = record["video_count"]
            result_tensor[artist_idx, 21] = 1
    logging.info("Saving artist tensor...")
    torch.save(result_tensor, "./tiktok_pyg/ds/artists.pt")
    logging.info("Artist tensor done")
    driver.close()


def build_track_tensor():
    track_map = get_track_map()
    logging.info("Building track tensor...")
    driver = GraphDatabase.driver(f"bolt://{DB_HOST}:{DB_PORT}", auth=basic_auth(DB_USER, DB_PASS))  # type: ignore
    result_tensor = torch.empty((len(track_map), 5 + 1), dtype=torch.float32)
    with driver.session() as session:
        query = """
            MATCH (n:Track)<-[:WORKED_IN]-(:Artist {tiktok_call: true})
            WITH DISTINCT n
            WITH
                n.id as id,
                n.popularity_scaled AS popularity_scaled,
                toInteger(n.year) AS year,
                toInteger(n.month) AS month
            RETURN
                id,
                popularity_scaled,
                case when year = 0 then 0 else 1 end as has_year,
                year,
                CASE WHEN month = 13 THEN 0 ELSE 1 END AS month_known,
                CASE WHEN month > 0 AND month <= 6 THEN 1 ELSE 0 END AS sem_1
            ;
        """
        q_result = session.run(query)
        for record in q_result:
            track_idx = track_map[record["id"]]
            result_tensor[track_idx, 0]  = record["popularity_scaled"]
            result_tensor[track_idx, 1]  = record["has_year"]
            result_tensor[track_idx, 2]  = record["year"]
            result_tensor[track_idx, 3]  = record["month_known"]
            result_tensor[track_idx, 4] = record["sem_1"]
            result_tensor[track_idx, 5] = 1
    logging.info("Saving track tensor...")
    torch.save(result_tensor, "./tiktok_pyg/ds/tracks.pt")
    logging.info("Track tensor done")
    driver.close()


def build_tag_tensor():
    tag_map = get_tag_map()
    logging.info("Building tag tensor...")
    result_tensor = torch.zeros((len(tag_map), len(tag_map) + 1), dtype=torch.float32)
    for i in range(len(tag_map)):
        result_tensor[i, i] = 1.0
    result_tensor[:, len(tag_map)] = 1.0
    torch.save(result_tensor, "./tiktok_pyg/ds/tags.pt")
    logging.info("Tag tensor done")


def build_worked_in_by_tensor():
    artist_map = get_artist_map()
    track_map = get_track_map()
    logging.info("Building worked_in and worked_by tensors...")
    driver = GraphDatabase.driver(f"bolt://{DB_HOST}:{DB_PORT}", auth=basic_auth(DB_USER, DB_PASS))  # type: ignore
    count = get_x_count("(:Artist {tiktok_call: true})-[:WORKED_IN]->(:Track)", driver)
    logging.info("Got %d records of WORKED_IN", count)
    result_tensor = torch.empty((2, count), dtype=torch.long)
    with driver.session() as session:
        query = """
            MATCH (n:Artist {tiktok_call: true})-[:WORKED_IN]->(m:Track)
            RETURN
                n.main_id AS artist_id,
                m.id AS track_id
            ;
        """
        q_result = session.run(query)
        for i, record in enumerate(q_result):
            artist_idx = artist_map[record["artist_id"]]
            track_idx = track_map[record["track_id"]]
            result_tensor[0, i] = artist_idx
            result_tensor[1, i] = track_idx
    logging.info("Saving worked_in tensor...")
    torch.save(result_tensor, "./tiktok_pyg/ds/worked_in.pt")
    logging.info("Saving worked_by tensor...")
    result_tensor[[0, 1], :] = result_tensor[[1, 0], :]
    torch.save(result_tensor, "./tiktok_pyg/ds/worked_by.pt")
    logging.info("worked_in and worked_by tensors done")
    driver.close()


def build_collab_with_tensor():
    artist_map = get_artist_map()
    logging.info("Building collab_with tensor...")
    driver = GraphDatabase.driver(f"bolt://{DB_HOST}:{DB_PORT}", auth=basic_auth(DB_USER, DB_PASS))  # type: ignore
    count = get_x_count("(:Artist {tiktok_call: true})-[:COLLAB_WITH]->(:Artist {tiktok_call: true})", driver)
    logging.info("Got %d records of COLLAB_WITH", count)
    result_tensor = torch.empty((2, count), dtype=torch.long)
    attr_tensor = torch.empty((count, 1), dtype=torch.float32)
    with driver.session() as session:
        query = """
            MATCH (n:Artist {tiktok_call: true})-[r:COLLAB_WITH]->(m:Artist {tiktok_call: true})
            WHERE n < m
            RETURN
                n.main_id AS artist0_id,
                m.main_id AS artist1_id,
                r.count as count
            ;
        """
        q_result = session.run(query)
        i = 0
        for record in q_result:
            artist0_idx = artist_map[record["artist0_id"]]
            artist1_idx = artist_map[record["artist1_id"]]
            count = record["count"]
            result_tensor[0, i] = artist0_idx
            result_tensor[1, i] = artist1_idx
            attr_tensor[i] = count
            result_tensor[0, i + 1] = artist1_idx
            result_tensor[1, i + 1] = artist0_idx
            attr_tensor[i + 1] = count
            i += 2
    logging.info("Saving collab_with tensor...")
    torch.save(result_tensor, "./tiktok_pyg/ds/collab_with.pt")
    logging.info("Saving collab_with attr tensor...")
    torch.save(attr_tensor, "./tiktok_pyg/ds/collab_with_attr.pt")
    logging.info("collab_with tensor done")
    driver.close()


def build_musically_related_to_tensor():
    artist_map = get_artist_map()
    logging.info("Building musically_related_to tensor...")
    driver = GraphDatabase.driver(f"bolt://{DB_HOST}:{DB_PORT}", auth=basic_auth(DB_USER, DB_PASS))  # type: ignore
    count = get_x_count("(:Artist {tiktok_call: true})-[:MUSICALLY_RELATED_TO]->(:Artist {tiktok_call: true})", driver)
    logging.info("Got %d records of MUSICALLY_RELATED_TO", count)
    result_tensor = torch.empty((2, count), dtype=torch.long)
    attr_tensor = torch.empty((count, 1), dtype=torch.float32)
    with driver.session() as session:
        query = """
            MATCH (n:Artist {tiktok_call: true})-[r:MUSICALLY_RELATED_TO]->(m:Artist {tiktok_call: true})
            WHERE n < m
            RETURN
                n.main_id AS artist0_id,
                m.main_id AS artist1_id,
                r.count as count
            ;
        """
        q_result = session.run(query)
        i = 0
        for record in q_result:
            artist0_idx = artist_map[record["artist0_id"]]
            artist1_idx = artist_map[record["artist1_id"]]
            count = record["count"]
            result_tensor[0, i] = artist0_idx
            result_tensor[1, i] = artist1_idx
            attr_tensor[i] = count
            result_tensor[0, i + 1] = artist1_idx
            result_tensor[1, i + 1] = artist0_idx
            attr_tensor[i + 1] = count
            i += 2
    logging.info("Saving musically_related_to tensor...")
    torch.save(result_tensor, "./tiktok_pyg/ds/musically_related_to.pt")
    logging.info("Saving musically_related_to attr tensor...")
    torch.save(attr_tensor, "./tiktok_pyg/ds/musically_related_to_attr.pt")
    logging.info("musically_related_to tensor done")
    driver.close()


def build_personally_related_to_tensor():
    artist_map = get_artist_map()
    logging.info("Building personally_related_to tensor...")
    driver = GraphDatabase.driver(f"bolt://{DB_HOST}:{DB_PORT}", auth=basic_auth(DB_USER, DB_PASS))  # type: ignore
    count = get_x_count("(:Artist {tiktok_call: true})-[:PERSONALLY_RELATED_TO]->(:Artist {tiktok_call: true})", driver)
    logging.info("Got %d records of PERSONALLY_RELATED_TO", count)
    result_tensor = torch.empty((2, count), dtype=torch.long)
    attr_tensor = torch.empty((count, 1), dtype=torch.float32)
    with driver.session() as session:
        query = """
            MATCH (n:Artist {tiktok_call: true})-[r:PERSONALLY_RELATED_TO]->(m:Artist {tiktok_call: true})
            WHERE n < m
            RETURN
                n.main_id AS artist0_id,
                m.main_id AS artist1_id,
                r.count as count
            ;
        """
        q_result = session.run(query)
        i = 0
        for record in q_result:
            artist0_idx = artist_map[record["artist0_id"]]
            artist1_idx = artist_map[record["artist1_id"]]
            count = record["count"]
            result_tensor[0, i] = artist0_idx
            result_tensor[1, i] = artist1_idx
            attr_tensor[i] = count
            result_tensor[0, i + 1] = artist1_idx
            result_tensor[1, i + 1] = artist0_idx
            attr_tensor[i + 1] = count
            i += 2
    logging.info("Saving personally_related_to tensor...")
    torch.save(result_tensor, "./tiktok_pyg/ds/personally_related_to.pt")
    logging.info("Saving personally_related_to attr tensor...")
    torch.save(attr_tensor, "./tiktok_pyg/ds/personally_related_to_attr.pt")
    logging.info("personally_related_to tensor done")
    driver.close()


def build_linked_to_tensor():
    artist_map = get_artist_map()
    logging.info("Building linked_to tensor...")
    driver = GraphDatabase.driver(f"bolt://{DB_HOST}:{DB_PORT}", auth=basic_auth(DB_USER, DB_PASS))  # type: ignore
    count = get_x_count("(:Artist {tiktok_call: true})-[:LINKED_TO]->(:Artist {tiktok_call: true})", driver)
    logging.info("Got %d records of LINKED_TO", count)
    result_tensor = torch.empty((2, count), dtype=torch.long)
    attr_tensor = torch.empty((count, 1), dtype=torch.float32)
    with driver.session() as session:
        query = """
            MATCH (n:Artist {tiktok_call: true})-[r:LINKED_TO]->(m:Artist {tiktok_call: true})
            WHERE n < m
            RETURN
                n.main_id AS artist0_id,
                m.main_id AS artist1_id,
                r.count as count
            ;
        """
        q_result = session.run(query)
        i = 0
        for record in q_result:
            artist0_idx = artist_map[record["artist0_id"]]
            artist1_idx = artist_map[record["artist1_id"]]
            count = record["count"]
            result_tensor[0, i] = artist0_idx
            result_tensor[1, i] = artist1_idx
            attr_tensor[i] = count
            result_tensor[0, i + 1] = artist1_idx
            result_tensor[1, i + 1] = artist0_idx
            attr_tensor[i + 1] = count
            i += 2
    logging.info("Saving linked_to tensor...")
    torch.save(result_tensor, "./tiktok_pyg/ds/linked_to.pt")
    logging.info("Saving linked_to attr tensor...")
    torch.save(attr_tensor, "./tiktok_pyg/ds/linked_to_attr.pt")
    logging.info("linked_to tensor done")
    driver.close()


def build_last_fm_match_tensor():
    artist_map = get_artist_map()
    logging.info("Building last_fm_match tensor...")
    driver = GraphDatabase.driver(f"bolt://{DB_HOST}:{DB_PORT}", auth=basic_auth(DB_USER, DB_PASS))  # type: ignore
    count = get_x_count("(:Artist {tiktok_call: true})-[:LAST_FM_MATCH]->(:Artist {tiktok_call: true})", driver)
    logging.info("Got %d records of LAST_FM_MATCH", count)
    result_tensor = torch.empty((2, count), dtype=torch.long)
    attr_tensor = torch.empty((count, 1), dtype=torch.float32)
    with driver.session() as session:
        query = """
            MATCH (n:Artist {tiktok_call: true})-[r:LAST_FM_MATCH]->(m:Artist {tiktok_call: true})
            WHERE n < m
            RETURN
                n.main_id AS artist0_id,
                m.main_id AS artist1_id,
                r.weight as weight
            ;
        """
        q_result = session.run(query)
        i = 0
        for record in q_result:
            artist0_idx = artist_map[record["artist0_id"]]
            artist1_idx = artist_map[record["artist1_id"]]
            weight = record["weight"]
            result_tensor[0, i] = artist0_idx
            result_tensor[1, i] = artist1_idx
            attr_tensor[i] = weight
            result_tensor[0, i + 1] = artist1_idx
            result_tensor[1, i + 1] = artist0_idx
            attr_tensor[i + 1] = weight
            i += 2
    logging.info("Saving last_fm_match tensor...")
    torch.save(result_tensor, "./tiktok_pyg/ds/last_fm_match.pt")
    logging.info("Saving last_fm_match attr tensor...")
    torch.save(attr_tensor, "./tiktok_pyg/ds/last_fm_match_attr.pt")
    logging.info("last_fm_match tensor done")
    driver.close()


def build_follows_tensor():
    artist_map = get_artist_map()
    logging.info("Building follows tensor...")
    driver = GraphDatabase.driver(f"bolt://{DB_HOST}:{DB_PORT}", auth=basic_auth(DB_USER, DB_PASS))  # type: ignore
    count = get_x_count("(:Artist {tiktok_call: true})-[:FOLLOWS]->(:Artist {tiktok_call: true})", driver)
    logging.info("Got %d records of FOLLOWS", count)
    result_tensor = torch.empty((2, count), dtype=torch.long)
    attr_tensor = torch.empty((count, 1), dtype=torch.float32)
    with driver.session() as session:
        query = """
            MATCH (n:Artist {tiktok_call: true})-[r:FOLLOWS]->(m:Artist {tiktok_call: true})
            RETURN
                n.main_id AS artist0_id,
                m.main_id AS artist1_id,
                r.weight as weight
            ;
        """
        q_result = session.run(query)
        i = 0
        for record in q_result:
            artist0_idx = artist_map[record["artist0_id"]]
            artist1_idx = artist_map[record["artist1_id"]]
            weight = record["weight"]
            result_tensor[0, i] = artist0_idx
            result_tensor[1, i] = artist1_idx
            attr_tensor[i] = weight
            i += 1
    logging.info("Saving follows tensor...")
    torch.save(result_tensor, "./tiktok_pyg/ds/follows.pt")
    logging.info("Saving follows attr tensor...")
    torch.save(attr_tensor, "./tiktok_pyg/ds/follows_attr.pt")
    logging.info("follows tensor done")
    driver.close()


def build_tags_has_tag_tensor_artists():
    artist_map = get_artist_map()
    tag_map = get_tag_map()
    logging.info("Building tags and has_tag tensors for artists...")
    driver = GraphDatabase.driver(f"bolt://{DB_HOST}:{DB_PORT}", auth=basic_auth(DB_USER, DB_PASS))  # type: ignore
    count = get_x_count("()-[:TAGS]->(:Artist {tiktok_call: true})", driver)
    logging.info("Got %d records of TAGS", count)
    result_tensor = torch.empty((2, count), dtype=torch.long)
    with driver.session() as session:
        query = """
            MATCH (n:Tag)-[:TAGS]->(m:Artist {tiktok_call: true})
            RETURN
                n.id AS tag_id,
                m.main_id AS artist_id
            ;
        """
        q_result = session.run(query)
        for i, record in enumerate(q_result):
            tag_idx = tag_map[record["tag_id"]]
            artist_idx = artist_map[record["artist_id"]]
            result_tensor[0, i] = tag_idx
            result_tensor[1, i] = artist_idx
    logging.info("Saving tags tensor for artists...")
    torch.save(result_tensor, "./tiktok_pyg/ds/tags_artists.pt")
    logging.info("Saving has_tag tensor for artists...")
    result_tensor[[0, 1], :] = result_tensor[[1, 0], :]
    torch.save(result_tensor, "./tiktok_pyg/ds/has_tag_artists.pt")
    logging.info("tags and has_tag tensors for artists done")
    driver.close()


def build_tags_has_tag_tensor_tracks():
    track_map = get_track_map()
    tag_map = get_tag_map()
    logging.info("Building tags and has_tag tensors for tracks...")
    driver = GraphDatabase.driver(f"bolt://{DB_HOST}:{DB_PORT}", auth=basic_auth(DB_USER, DB_PASS))  # type: ignore
    with driver.session() as session:
        count_query = """
            MATCH (:Artist {tiktok_call: true})-[:WORKED_IN]->(m:Track)
            WITH DISTINCT m
            MATCH (n:Tag)-[:TAGS]->(m)
            RETURN count(*) as c
        """
        count = session.run(count_query).data()[0]["c"]
        logging.info("Got %d records of TAGS", count)
        result_tensor = torch.empty((2, count), dtype=torch.long)
        query = """
            MATCH (:Artist {tiktok_call: true})-[:WORKED_IN]->(m:Track)
            WITH DISTINCT m
            MATCH (n:Tag)-[:TAGS]->(m)
            RETURN
                n.id AS tag_id,
                m.id AS track_id
            ;
        """
        q_result = session.run(query)
        for i, record in enumerate(q_result):
            tag_idx = tag_map[record["tag_id"]]
            track_idx = track_map[record["track_id"]]
            result_tensor[0, i] = tag_idx
            result_tensor[1, i] = track_idx
    logging.info("Saving tags tensor for tracks...")
    torch.save(result_tensor, "./tiktok_pyg/ds/tags_tracks.pt")
    logging.info("Saving has_tag tensor for tracks...")
    result_tensor[[0, 1], :] = result_tensor[[1, 0], :]
    torch.save(result_tensor, "./tiktok_pyg/ds/has_tag_tracks.pt")
    logging.info("tags and has_tag tensors for tracks done")
    driver.close()


def multiprocess_stuff(*jobs: multiprocessing.Process):
    for job in jobs:
        job.start()
    for job in jobs:
        job.join()


def main():
    multiprocess_stuff(
        multiprocessing.Process(target=build_artist_map),
        multiprocessing.Process(target=build_track_map),
        multiprocessing.Process(target=build_tag_map),
    )

    multiprocess_stuff(
        multiprocessing.Process(target=build_artist_tensor),
        multiprocessing.Process(target=build_track_tensor),
        multiprocessing.Process(target=build_tag_tensor),
        multiprocessing.Process(target=build_worked_in_by_tensor),
        multiprocessing.Process(target=build_collab_with_tensor),
        multiprocessing.Process(target=build_musically_related_to_tensor),
        multiprocessing.Process(target=build_personally_related_to_tensor),
        multiprocessing.Process(target=build_linked_to_tensor),
        multiprocessing.Process(target=build_last_fm_match_tensor),
        multiprocessing.Process(target=build_follows_tensor),
        multiprocessing.Process(target=build_tags_has_tag_tensor_artists),
        multiprocessing.Process(target=build_tags_has_tag_tensor_tracks),
    )


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

    if not os.path.exists("./tiktok_pyg/ds/"):
        os.mkdir("./tiktok_pyg/ds/")

    main()
