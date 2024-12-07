from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
from torch.nn.functional import normalize

from database.tables import StreamingEvents
from database.utils import PGManager

device = "cpu"


def preprocess_ms_played(log=True):
    db = PGManager()
    with db.get_session() as session:
        ms_played = (
            session.query(StreamingEvents.ms_played)
            .filter(StreamingEvents.episode_show_name == "Stuff You Should Know")
            .all()
        )

    ms_played = [x[0] for x in ms_played]
    ms_played = torch.tensor(ms_played, dtype=torch.float32).to(device)
    ms_played = normalize(ms_played, dim=0)
    ms_played = ms_played.unsqueeze(1)
    labels = ms_played.float()
    if log:
        labels = torch.log1p(labels)
    return labels


def preprocess_ts():
    db = PGManager()
    with db.get_session() as session:
        ts_values = (
            session.query(StreamingEvents.ts)
            .filter(StreamingEvents.episode_show_name == "Stuff You Should Know")
            .all()
        )

    ts_date_year = [x[0].year for x in ts_values]
    ts_date_month = [x[0].month for x in ts_values]
    ts_date_day_of_year_sin = [
        np.sin(2 * np.pi * x[0].timetuple().tm_yday / 365) for x in ts_values
    ]
    ts_date_day_of_year_cos = [
        np.cos(2 * np.pi * x[0].timetuple().tm_yday / 365) for x in ts_values
    ]

    ts_date_year = torch.tensor(ts_date_year, dtype=torch.float32).to(device)
    ts_date_month = torch.tensor(ts_date_month, dtype=torch.float32).to(device)
    ts_date_day_of_year_sin = (
        torch.tensor(ts_date_day_of_year_sin, dtype=torch.float32)
        .to(device)
        .unsqueeze(1)
    )
    ts_date_day_of_year_cos = (
        torch.tensor(ts_date_day_of_year_cos, dtype=torch.float32)
        .to(device)
        .unsqueeze(1)
    )
    ts_date_year_norm = normalize(ts_date_year, dim=0).to(device).unsqueeze(1)
    ts_date_year_month_norm = normalize(ts_date_month, dim=0).to(device).unsqueeze(1)
    return (
        ts_date_year_norm,
        ts_date_year_month_norm,
        ts_date_day_of_year_sin,
        ts_date_day_of_year_cos,
    )


def preprocess_conn_country():
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    db = PGManager()
    with db.get_session() as session:
        conn_country = (
            session.query(StreamingEvents.conn_country)
            .filter(StreamingEvents.episode_show_name == "Stuff You Should Know")
            .all()
        )

    conn_country = [x[0] for x in conn_country]
    conn_country_ohe = ohe.fit_transform(
        np.array(conn_country).reshape(-1, 1),
    )
    conn_country_ohe = torch.tensor(conn_country_ohe, dtype=torch.float32).to(device)
    conn_country_ohe = normalize(conn_country_ohe, dim=0)
    return conn_country_ohe


def preprocess_reason_start():
    ohe = OneHotEncoder(handle_unknown="ignore")
    db = PGManager()
    with db.get_session() as session:
        rs = (
            session.query(StreamingEvents.reason_start)
            .filter(StreamingEvents.episode_show_name == "Stuff You Should Know")
            .all()
        )
        rs = [x0 for (x0,) in rs]
        rs_ohe = ohe.fit_transform(np.array(rs).reshape(-1, 1))
        rs_ohe = torch.sparse_coo_tensor(rs_ohe.nonzero(), rs_ohe.data, rs_ohe.shape)
        rs_ohe.to(device)
        return rs_ohe


def preprocess_reason_end():
    ohe = OneHotEncoder(handle_unknown="ignore")
    db = PGManager()
    with db.get_session() as session:
        re = (
            session.query(StreamingEvents.reason_end)
            .filter(StreamingEvents.episode_show_name == "Stuff You Should Know")
            .all()
        )
    re = [x0 for (x0,) in re]
    re_ohe = ohe.fit_transform(np.array(re).reshape(-1, 1))
    re_ohe = torch.sparse_coo_tensor(re_ohe.nonzero(), re_ohe.data, re_ohe.shape)
    re_ohe.to(device)
    return re_ohe


def preprocess_skipped():
    db = PGManager()
    with db.get_session() as session:
        skipped = (
            session.query(StreamingEvents.skipped)
            .filter(StreamingEvents.episode_show_name == "Stuff You Should Know")
            .all()
        )
    skipped = [int(x0) for (x0,) in skipped]
    skipped = torch.tensor(skipped, dtype=torch.float32).to(device).unsqueeze(1)
    return skipped


def concat_user_features():
    (
        ts_date_year_norm,
        ts_date_year_month_norm,
        ts_date_day_of_year_sin,
        ts_date_day_of_year_cos,
    ) = preprocess_ts()
    conn_country_ohe_tensor = preprocess_conn_country()
    rs_ohe = preprocess_reason_start()
    re_ohe = preprocess_reason_end()
    skipped = preprocess_skipped()

    tensors_to_cat = [
        ts_date_year_norm,
        ts_date_year_month_norm,
        ts_date_day_of_year_sin,
        ts_date_day_of_year_cos,
        conn_country_ohe_tensor,
        rs_ohe,
        re_ohe,
        skipped,
    ]
    tensors_to_cat = [x.to_dense() for x in tensors_to_cat]
    pt_concat = torch.cat(
        (tensors_to_cat),
        dim=1,
    )
    return pt_concat
