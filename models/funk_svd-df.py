import time
from traceback import print_tb
import dask.array as da
import numpy as np
import sys
import numpy as np
from dask.distributed import Client, LocalCluster
from dask.distributed import wait
import dask.dataframe as dd
from numpy.core.fromnumeric import product
import pandas as pd
from dask import delayed, compute

class FunkSVD:
  def __init__(self, dask_client: Client, partition_size=100):
    self.dask_client = dask_client
    self.partition_size = partition_size * (10**6) #mb

  def __repartition_df(self, df: dd.DataFrame):
    self.n_partitions = 1 + df.memory_usage(deep=True).sum().compute() // self.partition_size
    df = df.repartition(npartitions=self.n_partitions)
    df = df.reset_index(drop=True)

    return df

  def fit(self,
          n_factors,
          train_df,
          user_col="user",
          item_col="item",
          rating_col="rating",
          epochs=50,
          lr=0.005,
          reg=0.02,
          val_split=None
  ):
    df_len = train_df.shape[0].compute()
    print(df_len)
    train_df = train_df.repartition(npartitions=1)
    train_df = train_df.reset_index(drop=True)

    # Data initialization
    u_ids = train_df[user_col].unique()
    i_ids = train_df[item_col].unique()
  
    self.u_mapping = { x: i for i, x in enumerate(u_ids) }
    self.i_mapping = { x: i for i, x in enumerate(i_ids) }
    train_df['u_encodings'] = train_df[user_col].map(self.u_mapping)
    train_df['i_encodings'] = train_df[item_col].map(self.i_mapping)

    self.n_users = len(u_ids)
    self.n_items = len(i_ids)

    print(self.n_users)
    print(self.n_items)

    self.min_rating = np.min(train_df[rating_col])
    self.max_rating = np.max(train_df[rating_col])
    self.mean_rating = np.mean(train_df[rating_col])

    u_factors_cols = ["u_f{}".format(i) for i in range(n_factors)]
    i_factors_cols = ["i_f{}".format(i) for i in range(n_factors)]

    train_df = train_df[["u_encodings", "i_encodings", rating_col]]

    # Initialize user & item biases.
    train_df["u_biases"] = 0
    train_df["i_biases"] = 0

    # Initialize random latent factors.
    u_factors = dd.from_dask_array(
      da.random.normal(0, .1, (df_len, n_factors)),
      columns=u_factors_cols
    ).repartition(npartitions=1)
    u_factors = u_factors.reset_index(1)

    # u_factors = dd.from_pandas(pd.Series(u_factors.values.compute().tolist()), npartitions=1)
    # u_factors = u_factors.reset_index(1)
    # train_df['u_factors'] = u_factors
    # del u_factors

    i_factors = dd.from_dask_array(
      da.random.normal(0, .1, (df_len, n_factors)),
      columns=i_factors_cols
    ).repartition(npartitions=1)
    i_factors = i_factors.reset_index(drop=True)

    # i_factors = dd.from_pandas(pd.Series(i_factors.values.compute().tolist()), npartitions=1)
    # i_factors = i_factors.reset_index(1)
    # train_df['i_factors'] = i_factors
    # del i_factors

    train_df = dd.concat([
        train_df,
        u_factors,
        i_factors
    ], axis=1, ignore_unknown_divisions=True)


    # Add errors column.
    train_df["error"] = 0
    train_df["u_mean_error"] = 0
    train_df["i_mean_error"] = 0

    # Repartition the df to chosen partition size.
    train_df = self.__repartition_df(train_df)
    # train_df = train_df.set_index("u_encodings")

    for epoch in range(epochs):
      print("Epoch: {}/{}".format(epoch + 1, epochs))
      epoch_start = time.time()

      def calculate_error(df, rating_col, mean_rating, n_factors):
        df["error"] = df["u_biases"] + df["i_biases"] + mean_rating

        for i in range(n_factors):
          df["error"] = df["u_f{}".format(i)] * df["i_f{}".format(i)]


        df["error"] = df[rating_col] - df["error"]
        return df

      #Calculate error on every partition of the df.
      # train_df = train_df.map_partitions(calculate_error, rating_col, self.mean_rating, n_factors)


      train_df["error"] = train_df["u_biases"] + train_df["i_biases"] + self.mean_rating

      for i in range(n_factors):
        train_df["error"] = train_df["u_f{}".format(i)] * train_df["i_f{}".format(i)]

      train_df["error"] = train_df[rating_col] - train_df["error"]

      # u_mean_error = train_df.groupby("u_encodings")["error"].mean().compute()
      # i_mean_error = train_df.groupby("i_encodings")["error"].mean().compute()

      # def update_biases(df, u_mean_error, i_mean_error, lr, reg):
      #   df["u_mean_error"] = df["u_encodings"].map(u_mean_error)
      #   df["i_mean_error"] = df["i_encodings"].map(i_mean_error)

      #   df["u_biases"] = lr * (df["u_mean_error"] - reg * df["u_biases"])
      #   df["i_biases"] = lr * (df["i_mean_error"] - reg * df["i_biases"])

      #   return df

      # train_df = train_df.map_partitions(update_biases, u_mean_error, i_mean_error, lr, reg)

      u_mean_error = train_df.groupby("u_encodings")["error"].mean(split_out=self.n_partitions)
      train_df["u_mean_error"] = train_df["u_encodings"].map(u_mean_error)
      train_df["u_biases"] = lr * (train_df["u_mean_error"] - reg * train_df["u_biases"])
      del u_mean_error


      # # Update the item biases.
      i_mean_error = train_df.groupby("i_encodings")["error"].mean(split_out=self.n_partitions)
      train_df['i_mean_error'] = train_df["i_encodings"].map(i_mean_error)
      train_df["i_biases"] = lr * (train_df["i_mean_error"] - reg * train_df["i_biases"])
      del i_mean_error


      # def update_biases(df, lr, reg):
      #   df["u_biases"] = lr * (df["u_mean_error"] - reg * df["u_biases"])
      #   df["i_biases"] = lr * (df["i_mean_error"] - reg * df["i_biases"])
      #   return df

      # train_df = train_df.map_partitions(update_biases, lr, reg)

      u_factor_mean = train_df.groupby("u_encodings")[u_factors_cols].mean(split_out=self.n_partitions)
      i_factor_mean = train_df.groupby("i_encodings")[i_factors_cols].mean(split_out=self.n_partitions)

      for i in range(n_factors):
        u_f = "u_f{}".format(i)
        i_f = "i_f{}".format(i)

        train_df[u_f] = train_df["u_encodings"].map(u_factor_mean[u_f])
        train_df[i_f] = train_df["i_encodings"].map(i_factor_mean[i_f])

        train_df[u_f] += lr * (train_df["u_mean_error"] * train_df[i_f] - reg * train_df[u_f])
        train_df[i_f] += lr * (train_df["i_mean_error"] * train_df[u_f] - reg * train_df[i_f])

      del u_factor_mean
      del i_factor_mean

      train_df["error"] = 0
      train_df["u_mean_error"] = 0
      train_df["i_mean_error"] = 0
      print(time.time() - epoch_start)

    compute_start = time.time()
    train_df = train_df.compute()
    print("COMPUTE: ", time.time() - compute_start)

    self.u_biases = train_df["u_biases"]
    self.i_biases = train_df["i_biases"]
    self.u_factors = train_df[u_factors_cols]
    self.i_factors = train_df[i_factors_cols]

    print(self.u_biases)
    print(self.i_biases)
    print(self.u_factors)
    print(self.u_factors)


    # for epoch in range(epochs):
    #   print("Epoch: {}/{}".format(epoch + 1, epochs))
    #   epoch_start_time = time.time()
      # x_len = x.shape[0]
      # for i in range(x_len):
      #   iteration_start_time = time.time()
      #   #self.print_epoch_status(i, x_len, "{} s".format(round(iteration_start_time - epoch_start_time, 3)))
      #   user, item, rating = int(x[i, 0]), int(x[i, 1]), x[i, 2]

      #   start_pred = self.mean_rating + u_biases[user] + i_biases[item]
      #   pred = 0
      #   # Predict rating
      #   # predictions = []
      #   for factor in range(n_factors):
      #       # product = delayed(lambda x, y: x * y)(u_factors[user, factor], i_factors[item, factor])
      #       # predictions.append(product)
      #       pred += u_factors[user, factor] * i_factors[item, factor]

      #   # pred = delayed(sum)(predictions).compute()
      #   # print(pred)

      #   pred += start_pred

      #   error = rating - pred

      #   # Update biases
      #   u_biases[user] += lr * (error - reg * u_biases[user])
      #   i_biases[item] += lr * (error - reg * i_biases[item])

      #   # Update latent factors
      #   for factor in range(n_factors):
      #       u_factor = u_factors[user, factor]
      #       i_factor = i_factors[item, factor]

      #       u_factors[user, factor] += lr * (error * i_factor - reg * u_factor)
      #       i_factors[item, factor] += lr * (error * u_factor - reg * i_factor)

    #   # print("\n")

    #   print(time.time() - epoch_start_time)

  def predict(self, test_df, user_col="user", item_col="item"):
    predictions = []
    user_c = 0
    for user, item in zip(test_df[user_col], test_df[item_col]):
      pred = self.mean_rating

      if user not in self.u_mapping or item not in self.i_mapping:
        user_c += 1
      else:
        u_id = self.u_mapping[user]
        i_id = self.i_mapping[item]
        pred += self.u_biases[u_id] + self.i_biases[i_id] + np.dot(self.u_factors[u_id], self.i_factors[i_id])
        pred = min(max(self.min_rating, pred), self.max_rating)

      predictions.append(pred)
    print("missing users: {}/{}".format(user_c, test_df.shape[0]))
    return predictions

  def eval(self, ground_truths, predictions):
    mae = self.mae(ground_truths, predictions)
    mse = self.mse(ground_truths, predictions)

    return mae, mse

  def mae(self, a, b):
    return (np.abs(np.subtract(a, b))).mean()
  
  def mse(self, a, b):
    return (np.square(np.subtract(a, b))).mean()

  def print_epoch_status(self, iter, max_iter, start_time, status):
    elapsed_time = time.time() - start_time 
    bar_length = 70
    j= iter / max_iter
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(bar_length * j):{bar_length}s}] {int(100 * j) + 1}% ({iter + 1}/{max_iter}) Elapsed time: {round(elapsed_time, 3)} s - {status}")
    sys.stdout.flush()

  def product(self, a, b):
    return a * b

  
def run():
    cluster = LocalCluster(n_workers=2)
    client = Client(cluster)

    df = pd.read_csv("data/small1.csv", names=["item", "user", "rating", "time"])
    df = df.drop_duplicates()
    # df = df.sort_values('time').drop_duplicates(subset=['item', 'user'], keep="last")
    df = df.drop('time', axis=1)

    # df = df[df['item'].isin(df['item'].value_counts()[df['item'].value_counts() > 2].index)]
    # df = df[df['user'].isin(df['user'].value_counts()[df['user'].value_counts() > 2].index)]

    df = dd.from_pandas(df, npartitions=1)
    train = df.sample(frac=1, random_state=7)

    model = FunkSVD(client, 100)
    model.fit(
        n_factors=20,
        train_df=train,
        epochs=2
    )

    client.shutdown()

if __name__ == '__main__':
    run()