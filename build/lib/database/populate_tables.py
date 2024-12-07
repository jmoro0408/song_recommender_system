import logging
import string
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from tables import Base, Episodes, StreamingEvents
from utils import PGManager
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_valid_uuid(uuid_string: str) -> bool:
    try:
        uuid_obj = uuid.UUID(uuid_string, version=4)
        return str(uuid_obj) == uuid_string
    except ValueError:
        return False


def clean_text(text: str | float) -> str | None:
    """
    Cleans the input text by removing punctuation and converting it to lowercase.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text.
    """
    if isinstance(text, float) or not text:
        return None
    text = text.replace("Selects: ", "").strip()
    text_no_punctuation = text.translate(str.maketrans("", "", string.punctuation))
    return text_no_punctuation.lower()


def insert_episodes() -> None:
    """
    Efficiently process each Parquet file that contains a single row
    and insert it into PostgreSQL, ensuring no duplicate episodes are added.
    """
    parquet_files = get_episode_paths()
    db = PGManager()
    inserted_episode_count = 0
    
    for file in tqdm(parquet_files, desc = "Inserting episodes"):
        logger.debug(f"Processing file: {file}")

        try:
            df = pd.read_parquet(file)
            df["name"] = "Stuff You Should Know"
            df = df.rename(
                columns={
                    "id": "episode_id",
                    "pubDate": "published_date",
                    "title": "episode_title",
                    "name": "podcast_name",
                }
            )
            df["clean_title"] = df["episode_title"].apply(clean_text)
            
            # Ensure episode_id is a valid UUID
            if not is_valid_uuid(df["episode_id"].iloc[0]):
                df["episode_id"] = str(uuid.uuid4())

            # Prepare the episode data for insertion
            episode_data = df[
                [
                    "episode_id",
                    "podcast_name",
                    "episode_title",
                    "link",
                    "summary",
                    "enc_len",
                    "transcript",
                    "published_date",
                    "clean_title",
                ]
            ].to_dict(orient="records")[0]

            with db.get_session() as session:
                # Check if episode with the same clean_title exists
                existing_episode = (
                    session.query(Episodes)
                    .filter_by(clean_title=episode_data["clean_title"])
                    .first()
                )

                if existing_episode:
                    logger.debug(
                        f"Duplicate episode found: {episode_data['clean_title']}. Skipping insertion."
                    )
                else:
                    # Insert the new episode
                    session.add(Episodes(**episode_data))
                    session.commit()
                    inserted_episode_count += 1
                    logger.debug(f"Inserted episode: {episode_data['clean_title']}")

        except Exception as e:
            logger.error(f"Error processing {file}: {e}")
        

    logger.info(f"Total inserted episodes: {inserted_episode_count}.")
    return None


def get_episode_paths() -> list[Path]:
    episode_data_dir = Path("src", "data", "episodes", "transcript_parquets")
    episodes_list = list(episode_data_dir.glob("*.parquet"))
    if len(episodes_list) == 0:
        raise Exception(f"No parquet files found in {episode_data_dir}")
    return episodes_list


def insert_listening() -> None:
    # TODO Refactor
    """
    This is a bit of a mess. A lot of edge cases and doing too much. 
    Ideally would refactor all the tables as well and organise better. 
    """
    inserted_streaming_event_count = 0
    listening_data_dir = Path(
        "src", "data", "listening", "extended_listening"
    )
    jsons = list(listening_data_dir.glob("*.json"))
    dfs = []
    for json in jsons:
        df = pd.read_json(json)
        dfs.append(df)
    df = pd.concat(dfs)
    input_dict = df.to_dict(orient="records")
    """Populate the StreamingEvents table with a list of dictionaries."""
    db = PGManager()
    with db.get_session() as session:
        for event in tqdm(input_dict, desc = "Inserting streaming events"):
            ts = datetime.strptime(event["ts"], "%Y-%m-%dT%H:%M:%SZ")
            ts = ts.strftime("%Y-%m-%d %H:%M:%S")
            ip_addr = event["ip_addr"].replace("x","255")
            streaming_clean_episode_name = clean_text(event["episode_name"])
            if (streaming_clean_episode_name 
                and (event["episode_show_name"] == "Stuff You Should Know")):
                # object is a SYSK podcast
                episode = (
                    session.query(Episodes)
                    .filter(Episodes.clean_title == streaming_clean_episode_name)
                    .first()
                )
                if episode: 
                    linked_episode_id = episode.episode_id
                else:
                    linked_episode_id = None 
                    #The parquet transcripts dont contain all sysk episodes so sometimes this is None
            else:
                linked_episode_id = None
            
            streaming_event = StreamingEvents(
                    ts=ts,
                    platform = event["platform"],
                    ms_played=event["ms_played"],
                    conn_country=event["conn_country"],
                    ip_addr=ip_addr,
                    master_metadata_track_name=event["master_metadata_track_name"],
                    master_metadata_album_artist_name=event["master_metadata_album_artist_name"],
                    master_metadata_album_album_name=event["master_metadata_album_album_name"],
                    spotify_track_uri=event["spotify_track_uri"],
                    episode_name=event["episode_name"],
                    episode_show_name=event["episode_show_name"],
                    spotify_episode_uri=event["spotify_episode_uri"],
                    reason_start=event["reason_start"],
                    reason_end=event["reason_end"],
                    shuffle=event["shuffle"],
                    skipped=event["skipped"],
                    episode_id=linked_episode_id 
                )
            session.add(streaming_event)
            inserted_streaming_event_count += 1
    logger.info(f"Inserted {inserted_streaming_event_count} streaming events.")

    return None


def main():
    db = PGManager()
    Base.metadata.bind = db.engine
    insert_episodes()
    insert_listening()


if __name__ == "__main__":
    main()
