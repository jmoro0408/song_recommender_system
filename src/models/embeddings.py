import logging

import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from sqlalchemy import select
from transformers import AutoModel, AutoTokenizer

from database.tables import Episodes
from database.utils import PGManager

load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BATCH_SIZE = 20
MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"  # embed dim = 1024


def pooling(outputs: torch.Tensor, inputs: dict, strategy: str = "cls") -> np.ndarray:
    # The model works really well with cls pooling (default) but also with mean pooling.
    if strategy == "cls":
        outputs = outputs[:, 0]
    elif strategy == "mean":
        outputs = torch.sum(
            outputs * inputs["attention_mask"][:, :, None], dim=1
        ) / torch.sum(inputs["attention_mask"], dim=1, keepdim=True)
    else:
        raise NotImplementedError
    return outputs.detach().cpu().numpy()


def embed_docs(docs: list[str]) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    inputs = tokenizer(docs, padding=True, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v
    outputs = model(**inputs).last_hidden_state
    return pooling(outputs, inputs, "cls")


def update_embeddings_in_batches():
    # This is not a very efficient way of doing this. But we're just doing it once so doesn't matter too much.
    db = PGManager()
    with db.get_session() as session:
        total_episodes = session.query(Episodes).count()
        print(f"Total episodes to process: {total_episodes}")

        for offset in range(0, total_episodes, BATCH_SIZE):
            episodes = (
                session.query(Episodes)
                .order_by(Episodes.episode_id)  # Ensure a consistent order
                .offset(offset)
                .limit(BATCH_SIZE)
                .all()
            )

            if not episodes:
                break

            summaries = [episode.summary for episode in episodes]
            embeddings = embed_docs(summaries)

            for episode, embedding in zip(episodes, embeddings):
                episode.summary_embedding = embedding

            session.commit()
            logger.info(
                f"Processed batch {offset // BATCH_SIZE + 1}/{-(-total_episodes // BATCH_SIZE)}"
            )


def get_closest_episode(episode_title: int) -> list[str]:
    db = PGManager()
    with db.get_session() as session:
        example_episode = (
            session.query(Episodes)
            .filter(Episodes.episode_title == episode_title)
            .first()
        )
        example_episode_embedding = example_episode.summary_embedding
        stmt = (
            select(Episodes)
            .order_by(Episodes.summary_embedding.l2_distance(example_episode_embedding))
            .limit(5)
        )

        result = session.execute(stmt).scalars().all()
        return [result.episode_title for result in result]


if __name__ == "__main__":
    update_embeddings_in_batches()
