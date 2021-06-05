import time
import dask.array as da
from dask.array.core import block
from dask.base import persist
import numpy as np
import sys
from dask.distributed import Client
import dask.dataframe as dd
import pandas as pd
from dask import compute
import sparse

class ALS:
  def __init__(self, client: Client):
    self.client = client

  def __preprocess_data(self, df, user_col, item_col, rating_col, chunk_size, n_factors):
    start_time = time.time()
    self.__print_status(0, 100, start_time, "Preprocessing data")
    self.chunk_size = chunk_size
    self.n_factors = n_factors
    self.user_col = user_col
    self.item_col = item_col
    self.rating_col = rating_col

    self.u_ids = df[user_col].unique()
    self.i_ids = df[item_col].unique()
    
    self.u_mapping = { x: i for i, x in enumerate(self.u_ids) }
    self.i_mapping = { x: i for i, x in enumerate(self.i_ids) }
    df['u_encodings'] = df[user_col].map(self.u_mapping)
    df['i_encodings'] = df[item_col].map(self.i_mapping)
    self.__print_status(50, 100, start_time, "Preprocessing data")

    self.n_users = len(self.u_ids)
    self.n_items = len(self.i_ids)
    self.n_ratings = df.shape[0]

    self.min_rating = np.min(df[rating_col])
    self.max_rating = np.max(df[rating_col])
    self.mean_rating = np.mean(df[rating_col])

    df = df[["u_encodings", "i_encodings", rating_col]]
    self.__print_status(99, 100, start_time, "Preprocessing data")
    print()
    return df

  def __create_sparse_chunked_matrix(self, df):
    start_time = time.time()
    df_val = df.values
    sparse_df = sparse.COO(df_val[:, :2].T.astype(int), df_val[:, 2], shape=((self.n_users, self.n_items)))

    # chunks = []
    # for i in range(0, self.n_users, self.chunk_size):
    #   sub_chunks=[]
    #   self.__print_status(i, self.n_users, start_time, "Creating sparse-chunked matrix")
    #   for j in range(0, self.n_items, self.chunk_size):
    #     sub_chunks.append(sparse_df[i: i + self.chunk_size, j: j + self.chunk_size])
    #   chunks.append(sub_chunks)


    # chunks = []
    # for i in range(0, self.n_items):
    #   chunks.append(sparse_df[0: self.n_users, i: i + 1])
      
    # x = da.block(chunks)
    # x_mask = da.sign(x).map_blocks(lambda x: x.todense(), dtype=np.ndarray) == 1

    # chunks = []
    # for i in range(0, self.n_users):
    #   chunks.append(sparse_df[i: i + 1, 0: self.n_items].T)
    
    # x_t = da.block(chunks)
    # x_mask_t = da.sign(x_t).map_blocks(lambda x: x.todense(), dtype=np.ndarray) == 1
      

    # x = sparse_df
    # x_mask = np.sign(x)

    # print()

    return x, x_mask, x_t, x_mask_t

  def __init_latent_vectors(self):
    # u_factors = da.random.random(0, 0.1, (self.n_users, self.n_factors), chunks=(self.chunk_size, self.n_factors))
    # i_factors = da.random.random(0, 0.1, (self.n_factors, self.n_items), chunks=(self.n_factors, self.chunk_size))

    # u_factors = np.random.rand(self.n_users, self.n_factors)
    # i_factors = np.random.rand(self.n_factors, self.n_items)

    u_factors = np.random.uniform(0, 0.1, (self.n_users, self.n_factors))
    i_factors = np.random.uniform(0, 0.1, (self.n_factors, self.n_items))

    return u_factors, i_factors

  def __get_training_errors(self, error):
    mae = np.sum(np.absolute(error)) / self.n_ratings
    mse = np.sum(error ** 2) / self.n_ratings
    rmse = np.sqrt(mse)

    return (mae, mse, rmse)

  def __plot_training_errors(self, errors):
    return

  def fit(self,
          n_factors,
          train_df,
          chunk_size,
          epochs=50,
          reg=0.02,
          collect_errors=False,
          user_col="user",
          item_col="item",
          rating_col="rating",
  ):
    df = self.__preprocess_data(train_df, user_col, item_col, rating_col, chunk_size, n_factors)
    x, x_mask, x_t, x_mask_t = self.__create_sparse_chunked_matrix(df)
    u_factors, i_factors = self.__init_latent_vectors()

    train_errors = []
    # x_compute = x.compute()
    # x_mask_compute = x_mask.compute()

    I = reg * np.eye(n_factors)
    start_time_epoch = time.time()
    print("starting epochs")
    for epoch in range(epochs):
      def compute_items(mask_row, x, u_factors, I, block_id=None):
        if block_id:
          masked_factors = u_factors * mask_row
          a = da.dot(masked_factors.T, u_factors) + I
          b = da.dot(masked_factors.T, x)
          return np.linalg.solve(a, b)
        return mask_row

      print("computing factors")
      i_factors = da.map_blocks(compute_items, x_mask, x, u_factors, I)
      i_factors = i_factors.compute()
      print(i_factors)

      # def compute_users(mask_col, x_t, i_factors, I, block_id=None):
      #   if block_id:
      #     masked_factors = i_factors * mask_col.T
      #     a = np.dot(i_factors, masked_factors.T) + I
      #     b = np.dot(masked_factors, x_t)
      #     return np.linalg.solve(a, b)
      #   return mask_col

      # u_factors = da.map_blocks(compute_users, x_mask_t, x_t, i_factors, I).T
      # u_factors = u_factors.compute()

      # if collect_errors:
      #   pred = np.dot(u_factors, i_factors)
      #   error = x_compute - pred * x_mask_compute
      #   train_errors.append(self.__get_training_errors(error))

      self.__print_status(epoch + 1, epochs, start_time_epoch, "Processing epoch", step=True)
    
    print(train_errors)

    self.u_factors = u_factors
    self.i_factors = i_factors
    print(self.u_factors)
    print(self.i_factors)

    self.x = np.dot(u_factors, i_factors)
 
  def predict(self, test_df, user_col=None, item_col=None):
    if user_col is None: user_col = self.user_col
    if item_col is None: item_col = self.item_col

    predictions = []
    start_time = time.time()
    df = test_df[[user_col, item_col]].values
    df_len = len(df)

    for i in range(df_len):
      user, item = df[i][0], df[i][1]
      self.__print_status(i, df_len, start_time, "Predicting...")
      pred = self.mean_rating

      if user in self.u_mapping and item in self.i_mapping:
        u_id = self.u_mapping[user]
        i_id = self.i_mapping[item]

        pred = self.x[u_id][i_id]
        pred = min(max(self.min_rating, pred), self.max_rating)
      
      predictions.append(pred)

    return predictions

  def eval(self, ground_truths, predictions):
    mae = self.__mae(ground_truths, predictions)
    mse = self.__mse(ground_truths, predictions)
    rmse = self.__rmse(ground_truths, predictions)

    return mae, mse, rmse

  def __mae(self, a, b):
    return np.abs(np.subtract(a, b)).mean()
  
  def __mse(self, a, b):
    return np.square(np.subtract(a, b)).mean()

  def __rmse(self, a, b):
    return np.sqrt(((np.subtract(a, b))**2).mean())

  def __print_status(self, iter, max_iter, start_time, status, step=False):
    elapsed_time = time.time() - start_time 
    bar_length = 70
    j= iter / max_iter
    sys.stdout.write('\r')
    if step:
      sys.stdout.write(f"[{'=' * int(bar_length * j):{bar_length}s}] {int(100 * j)}% ({iter}/{max_iter}) Elapsed time: {round(elapsed_time, 3)} s - {status}")
    else:
      sys.stdout.write(f"[{'=' * int(bar_length * j):{bar_length}s}] {int(100 * j) + 1}% Elapsed time: {round(elapsed_time, 3)} s - {status}")
    sys.stdout.flush()

def run():
    client = Client(n_workers=2)

    df = pd.read_csv("data/prime_pantry_5.csv", names=["item", "user", "rating", "time"])
    df = df.drop_duplicates()
    df = df.sort_values('time').drop_duplicates(subset=['item', 'user'], keep="last")
    df = df.drop('time', axis=1)
    df = df.head(50000)

    train = df.sample(frac=0.7, random_state=7)
    print(train.shape)
    test = df.drop(train.index.tolist())

    model = ALS(client)
    model.fit(
        n_factors=20,
        train_df=train,
        epochs=1,
        chunk_size=50,
        collect_errors=True
    )

    predictions = model.predict(test)
    # print(len(predictions))

    gt = test["rating"].to_numpy()
    # print(len(gt))

    print(model.eval(gt, predictions))

    client.shutdown()

if __name__ == '__main__':
    run()