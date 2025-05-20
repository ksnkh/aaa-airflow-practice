import os
import random
from datetime import timedelta
import argparse

import implicit
import mlflow
import numpy as np
import polars as pl
import threadpoolctl
from scipy.sparse import csr_matrix

threadpoolctl.threadpool_limits(1, "blas")
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

EVAL_DAYS_TRESHOLD = 14
DATA_DIR = 'data/'
SEED = 14

random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
pl.set_random_seed(SEED)


def get_data():
    df_test_users = pl.read_parquet(f'{DATA_DIR}/test_users.pq')
    df_clickstream = pl.read_parquet(f'{DATA_DIR}/clickstream.pq')
    df_event = pl.read_parquet(f'{DATA_DIR}/events.pq')
    return df_test_users, df_clickstream, df_event


def sample_train(df: pl.DataFrame, fraction: float = 1.0) -> pl.DataFrame:
    if fraction == 1.0:
        return df

    # Get one random row_id per group
    rep_ids = (
        df
        .group_by("cookie", maintain_order=True)
        .agg(pl.col("row_id").shuffle().first())
    )

    # Join back to get full rows for representatives
    representatives = df.join(rep_ids, on="row_id", how="inner").drop("cookie_right")

    # Exclude representatives from df
    non_representatives = df.join(rep_ids, on="row_id", how="anti")

    # Compute how many more rows to sample
    n_total = df.height
    n_needed = int(n_total * fraction)
    n_to_sample = n_needed - representatives.height

    if n_to_sample < 0:
        raise ValueError("Fraction too small to include one row per unique cookie.")

    sampled_rest = non_representatives.sample(n=n_to_sample, with_replacement=False, seed=SEED)

    final_sample = pl.concat([representatives, sampled_rest])

    return final_sample


def split_train_test(df_clickstream: pl.DataFrame, df_event: pl.DataFrame):
    treshhold = df_clickstream['event_date'].max() - timedelta(days=EVAL_DAYS_TRESHOLD)

    df_train = df_clickstream.filter(df_clickstream['event_date'] <= treshhold)
    df_eval = df_clickstream.filter(df_clickstream['event_date'] > treshhold)[['cookie', 'node', 'event']]

    df_eval = df_eval.join(df_train, on=['cookie', 'node'], how='anti')

    df_eval = df_eval.filter(
        pl.col('event').is_in(
            df_event.filter(pl.col('is_contact') == 1)['event'].unique()
        )
    )
    df_eval = df_eval.filter(
        pl.col('cookie').is_in(df_train['cookie'].unique())
    ).filter(
        pl.col('node').is_in(df_train['node'].unique())
    )

    df_eval = df_eval.unique(['cookie', 'node'])

    return df_train, df_eval


def get_als_pred(users, nodes, user_to_pred, params, confidence=None):
    user_ids = users.unique().to_list()
    item_ids = nodes.unique().to_list()

    user_id_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}

    index_to_item_id = {v: k for k, v in item_id_to_index.items()}

    rows = users.replace_strict(user_id_to_index).to_list()
    cols = nodes.replace_strict(item_id_to_index).to_list()

    if confidence is not None:
        values = [1 + params['als_conf_coef'] * conf for conf in confidence]
    else:
        values = [1] * len(users)

    sparse_matrix = csr_matrix((values, (rows, cols)), shape=(len(user_ids), len(item_ids)))

    model = implicit.als.AlternatingLeastSquares(iterations=params['als_iterations'],
                                                 factors=params['als_factors'],
                                                 regularization=params['als_regularization'],
                                                 alpha=params['als_alpha'],
                                                 random_state=14)
    model.fit(sparse_matrix)

    user4pred = np.array([user_id_to_index[i] for i in user_to_pred])

    recommendations, scores = model.recommend(user4pred,
                                              sparse_matrix[user4pred],
                                              N=40,
                                              filter_already_liked_items=True)

    df_pred = pl.DataFrame(
        {
            'node': [
                [index_to_item_id[i] for i in i] for i in recommendations.tolist()
            ],
            'cookie': list(user_to_pred),
            'scores': scores.tolist()
        }
    )
    df_pred = df_pred.explode(['node', 'scores'])
    return df_pred


def get_event_conf(train, event, run_params):
    confidence_expr = pl.when(
        pl.col("event").is_in(
            event.filter(pl.col('is_contact') == 1)['event'].unique()
        )
    ).then(1).otherwise(0)

    if run_params['als_data_prep'] == 'binary_wtime':
        treshhold = train['event_date'].max() - timedelta(days=EVAL_DAYS_TRESHOLD)
        days_since_expr = (treshhold - pl.col("event_date")).dt.total_days().cast(pl.Float64)

        time_decay_expr = (-(run_params['als_decay_rate']) * days_since_expr).exp()
        confidence_expr = confidence_expr * time_decay_expr

    event_conf = train.select(confidence_expr.alias("is_contact"))["is_contact"]

    return event_conf


def run_pipeline(df_train, df_test_users, df_event, run_params):
    users = df_train["cookie"]
    nodes = df_train["node"]
    test_users = df_test_users['cookie'].unique().to_list()

    conf = None
    if run_params['als_data_prep'] != 'baseline':
        conf = get_event_conf(df_train, df_event, run_params)

    df_pred = get_als_pred(users,
                           nodes,
                           test_users,
                           confidence=conf,
                           params=run_params)
    return df_pred


def run_experiment(df_train, df_eval, df_event, run_params):
    if run_params['data_frac'] == 1.0:
        cur_train = df_train
    else:
        cur_train = sample_train(df_train, run_params['data_frac'])

    with mlflow.start_run(run_name=run_params['run_name']):
        mlflow.log_params(run_params)

        df_pred = run_pipeline(cur_train, df_eval, df_event, run_params)

        score = recall_at(df_eval, df_pred, k=40)
        mlflow.log_metric("Recall_40", score)
        mlflow.log_dict(run_params, 'params.json')
    return score


def recall_at(df_true, df_pred, k=40):
    return df_true[['node', 'cookie']].join(
        df_pred.group_by('cookie').head(k).with_columns(value=1)[['node', 'cookie', 'value']],
        how='left',
        on=['cookie', 'node']
    ).select(
        [pl.col('value').fill_null(0), 'cookie']
    ).group_by(
        'cookie'
    ).agg(
        [
            pl.col('value').sum() / pl.col(
                'value'
            ).count()
        ]
    )['value'].mean()


def main():
    mlflow.set_tracking_uri(
        os.environ.get('MLFLOW_TRACKING_URI')
    )

    default_params = {
        'run_name': 'airflow',
        'als_data_prep': 'binary_wtime',
        'data_frac': 1,
        'model': 'als',
        'als_iterations': 5,
        'als_factors': 100,
        'als_regularization': 0.060000000000000005,
        'als_alpha': 3.8,
        'als_conf_coef': 5.341222514349242,
        'als_decay_rate': 0.029361182371878736}

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name')
    for param in default_params.keys():
        parser.add_argument(f'--{param}',
                            type=type(default_params[param]),
                            default=default_params[param])

    run_params = vars(parser.parse_args())

    EXPERIMENT_NAME = run_params.get('experiment_name')

    if not mlflow.get_experiment_by_name(EXPERIMENT_NAME):
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location='mlflow-artifacts:/')

    mlflow.set_experiment(EXPERIMENT_NAME)

    df_test_users, df_clickstream, df_event = get_data()
    df_train, df_eval = split_train_test(df_clickstream, df_event)

    run_experiment(df_train, df_eval, df_event, run_params)


if __name__ == '__main__':
    main()
