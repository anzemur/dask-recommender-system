import sys; sys.path.append(".")
import pandas as pd
import numpy as np
from dask.distributed import Client
import dask.dataframe as dd
from models import FunkSVD

def run():
    client = Client()

    df = pd.read_csv("data/small.csv", names=["item", "user", "rating", "time"])

    df = df.drop_duplicates()
    df = df.drop('time', axis=1)

    # df = df[df['item'].isin(df['item'].value_counts()[df['item'].value_counts() > 5].index)]
    # df = df[df['user'].isin(df['user'].value_counts()[df['user'].value_counts() > 5].index)]

    ddf = dd.from_pandas(df, npartitions=4).compute()

    train = ddf.sample(frac=0.8, random_state=7)
    print(train.head())



    # test = df.drop(train.index.tolist())
    # gt = test['rating']

    model = FunkSVD(client)
    model.fit(
        n_factors=100,
        train_df=train,
        epochs=2
    )

    # predictions = model.predict(test)
    # print(model.eval(gt, predictions))

    client.shutdown()

if __name__ == '__main__':
    run()