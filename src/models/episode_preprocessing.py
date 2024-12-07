import numpy as np
import torch
from torch.nn.functional import normalize

from database.tables import Episodes
from database.utils import PGManager


def preprocess_enc_len() -> torch.Tensor:
    db = PGManager()
    with db.get_session() as session:
        enc_len_values = session.query(Episodes.enc_len).all()

    enc_len_values = [x[0] for x in enc_len_values]
    enc_len_values = torch.tensor(enc_len_values, dtype=torch.float32)
    return normalize(enc_len_values, dim=0).unsqueeze(1)


def preprocess_published_date() -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    db = PGManager()
    with db.get_session() as session:
        published_date = session.query(Episodes.published_date).all()

    published_date_year = [x[0].year for x in published_date]
    published_date_month = [x[0].month for x in published_date]
    published_date_day_of_year_sin = [
        np.sin(2 * np.pi * x[0].timetuple().tm_yday / 365) for x in published_date
    ]
    published_date_day_of_year_cos = [
        np.cos(2 * np.pi * x[0].timetuple().tm_yday / 365) for x in published_date
    ]

    published_date_year = torch.tensor(published_date_year, dtype=torch.float32)
    published_date_month = torch.tensor(published_date_month, dtype=torch.float32)
    published_date_day_of_year_sin = torch.tensor(
        published_date_day_of_year_sin, dtype=torch.float32
    )
    published_date_day_of_year_cos = torch.tensor(
        published_date_day_of_year_cos, dtype=torch.float32
    )
    published_date_year_norm = normalize(published_date_year, dim=0)
    published_date_year_month_norm = normalize(published_date_month, dim=0)
    return (
        published_date_year_norm.unsqueeze(1),
        published_date_year_month_norm.unsqueeze(1),
        published_date_day_of_year_sin.unsqueeze(1),
        published_date_day_of_year_cos.unsqueeze(1),
    )


def get_summary_embeddings() -> torch.Tensor:
    db = PGManager()
    with db.get_session() as session:
        summary_embedding = session.query(Episodes.summary_embedding).all()

    summary_embedding = [x[0] for x in summary_embedding]
    return torch.tensor(summary_embedding, dtype=torch.float32)


def concat_episode_features() -> torch.Tensor:
    enc_len_values_norm = preprocess_enc_len()
    (
        published_date_year_norm,
        published_date_year_month_norm,
        published_date_day_of_year_sin,
        published_date_day_of_year_cos,
    ) = preprocess_published_date()
    summary_embeddings = get_summary_embeddings()

    embeddings_to_concat = [
        enc_len_values_norm,
        published_date_year_norm,
        published_date_year_month_norm,
        published_date_day_of_year_sin,
        published_date_day_of_year_cos,
        summary_embeddings,
    ]

    return torch.concat(embeddings_to_concat, dim=-1)
