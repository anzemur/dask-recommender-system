from dask.distributed import Client
import pandas as pd
from models import ALS

def run():
    client = Client(n_workers=3)

    df = pd.read_csv("data/musical_instruments_5.csv", names=["item", "user", "rating", "time"])
    df = df.drop_duplicates()
    df = df.sort_values('time').drop_duplicates(subset=['item', 'user'], keep="last")
    df = df.drop('time', axis=1)

    train = df.sample(frac=0.7, random_state=7)
    test = df.drop(train.index.tolist())

    model = ALS(client)
    model.fit(
        n_factors=30,
        train_df=train,
        epochs=100,
        chunk_size=4000,
        collect_errors=True,
        plot_errors=True
    )

    predictions = model.predict(test)
    gt = test["rating"].to_numpy()
    eval = model.eval(gt, predictions)
    print(eval)
    client.shutdown()

if __name__ == '__main__':
    run()