import time
import dask.array as da
import numpy as np
import sys
from dask.distributed import Client
from dask import compute
import sparse
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime

class ALS:
  def __init__(self, client: Client):
    self.client = client

  def __preprocess_data(self, df, user_col, item_col, rating_col, chunk_size, n_factors):
    start_time = time.time()
    self.__print_status(0, 100, start_time, "Preprocessing data...")
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
    self.__print_status(50, 100, start_time, "Preprocessing data...")

    self.n_users = len(self.u_ids)
    self.n_items = len(self.i_ids)
    self.n_ratings = df.shape[0]

    self.min_rating = np.min(df[rating_col])
    self.max_rating = np.max(df[rating_col])
    self.mean_rating = np.mean(df[rating_col])

    df = df[["u_encodings", "i_encodings", rating_col]]
    self.__print_status(100, 100, start_time, "Preprocessing data...")
    print()
    return df

  def __create_sparse_chunked_matrix(self, df):
    start_time = time.time()
    df_val = df.values
    sparse_df = sparse.COO(df_val[:, :2].T.astype(int), df_val[:, 2], shape=((self.n_users, self.n_items)))

    chunks = []
    for i in range(0, self.n_users, self.chunk_size):
      sub_chunks=[]
      self.__print_status(i + self.chunk_size, self.n_users, start_time, "Creating user-item matrix...")
      for j in range(0, self.n_items, self.chunk_size):
        sub_chunks.append(sparse_df[i: i + self.chunk_size, j: j + self.chunk_size])
      chunks.append(sub_chunks)

    self.__print_status(self.n_users, self.n_users, start_time, "Creating user-item matrix...")
    x = da.block(chunks)
    x_mask = da.sign(x).map_blocks(lambda x: x.todense(), dtype=np.ndarray) == 1
    print()

    return x, x_mask

  def __init_biases(self):
    u_biases = da.zeros((self.n_users, 1), chunks=(self.chunk_size,1))
    i_biases = da.zeros(self.n_items, chunks=(self.chunk_size,))
    return u_biases, i_biases

  def __init_latent_vectors(self):
    u_factors = da.random.uniform(0, 0.1, (self.n_users, self.n_factors), chunks=(self.chunk_size, self.n_factors))
    i_factors = da.random.uniform(0, 0.1, (self.n_items, self.n_factors), chunks=(self.chunk_size, self.n_factors))
    return u_factors, i_factors

  def __compute_learing_error(self, u_factors, i_factors, u_biases, i_biases, x, x_mask):
    pred = self.mean_rating + u_biases + u_factors @ i_factors.T + i_biases
    error = x - pred * x_mask
    return error

  def __get_training_errors(self, error):
    mae = da.sum(da.absolute(error)) / self.n_ratings
    mse = da.sum(error ** 2) / self.n_ratings
    rmse = da.sqrt(mse)
    return (mae, mse, rmse)

  def __plot_training_errors(self, errors):
    if not os.path.exists('res/'):
      os.mkdir('res/')

    mapped_errors = {
      "MAE": [],
      "MSE": [],
      "RMSE": [],
    }

    for error in errors:
      mapped_errors["MAE"].append(error[0])
      mapped_errors["MSE"].append(error[1])
      mapped_errors["RMSE"].append(error[2])

    sns.set_style("darkgrid")
    start_time = time.time()
    plt.figure()
    plt.subplots(figsize=(30, 5))
    for index, (error, error_values) in enumerate(mapped_errors.items()):
      self.__print_status(index + 1, len(mapped_errors), start_time, "Ploting training errors...")
      plt.subplot(130 + index + 1)
      plt.xlabel("Epoch", fontsize=24)
      plt.ylabel(error, fontsize=24)
      plt.plot(error_values)
    
    plt.savefig("res/{}-{}-{}.pdf".format(type(self).__name__, "training-errors", datetime.today().strftime('%Y-%m-%d-%H:%M:%S')))

    print()

  def __mae(self, a, b):
    return np.abs(np.subtract(a, b)).mean()
  
  def __mse(self, a, b):
    return np.square(np.subtract(a, b)).mean()

  def __rmse(self, a, b):
    return np.sqrt(((np.subtract(a, b))**2).mean())

  def __print_status(self, iter, max_iter, start_time, status, step=False):
    elapsed_time = time.time() - start_time 
    bar_length = 70
    j = iter / max_iter
    sys.stdout.write('\r')
    if step:
      sys.stdout.write(f"[{'=' * int(bar_length * j):{bar_length}s}] {int(100 * j)}% Elapsed time: {round(elapsed_time, 3)} s - {status} ({iter}/{max_iter})")
    else:
      sys.stdout.write(f"[{'=' * int(bar_length * j):{bar_length}s}] {int(100 * j)}% Elapsed time: {round(elapsed_time, 3)} s - {status}")
    sys.stdout.flush()

  def fit(self,
          n_factors,
          train_df,
          chunk_size,
          epochs=50,
          lr=0.001,
          reg=0.001,
          collect_errors=False,
          plot_errors=False,
          user_col="user",
          item_col="item",
          rating_col="rating",
  ):
    fit_start_time = time.time()
    df = self.__preprocess_data(train_df, user_col, item_col, rating_col, chunk_size, n_factors)
    x, x_mask = self.__create_sparse_chunked_matrix(df)
    u_biases, i_biases = self.__init_biases()
    u_factors, i_factors = self.__init_latent_vectors()

    start_time_epoch = time.time()
    train_errors = []
    error = self.__compute_learing_error(u_factors, i_factors, u_biases, i_biases, x, x_mask)
    for epoch in range(epochs):
      self.__print_status(epoch + 1, epochs, start_time_epoch, "Creating epochs", step=True)

      if collect_errors:
        train_errors.append(self.__get_training_errors(error))

      u_factors = u_factors + lr * (error @ i_factors - reg * u_factors)
      u_biases = u_biases + lr * da.sum(error - reg * u_biases, axis=1, keepdims=True)

      error = self.__compute_learing_error(u_factors, i_factors, u_biases, i_biases, x, x_mask)
      i_factors = i_factors + lr * ((u_factors.T @ error).T - reg * i_factors)
      i_biases = i_biases + lr * da.sum(error - reg * i_biases, axis=0, keepdims=True)

      error = self.__compute_learing_error(u_factors, i_factors, u_biases, i_biases, x, x_mask)

    print("\nComputing in parallel...")

    compute_start_time = time.time()
    if collect_errors:
      self.u_biases, self.i_biases, self.u_factors, self.i_factors, self.train_errors = compute(u_biases, i_biases, u_factors, i_factors, train_errors)
    else:
      self.u_biases, self.i_biases, self.u_factors, self.i_factors = compute(u_biases, i_biases, u_factors, i_factors)
    self.u_biases = self.u_biases.T
    compute_end_time = time.time()

    print("Compute parallel time: {} s".format(round(compute_end_time - compute_start_time, 3)))
    print("Compute parallel time per epoch: {} s".format(round((compute_end_time - compute_start_time) / epochs, 3)))
    print("Total fitting time: {} s".format(round(compute_end_time - fit_start_time, 3)))

    if plot_errors:
      self.__plot_training_errors(self.train_errors)
 
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

        pred += self.u_biases[0][u_id] + self.i_biases[0][i_id] + self.u_factors[u_id] @ self.i_factors[i_id]
        pred = min(max(self.min_rating, pred), self.max_rating)
      
      predictions.append(pred)
    print()

    return predictions

  def eval(self, ground_truths, predictions):
    mae = self.__mae(ground_truths, predictions)
    mse = self.__mse(ground_truths, predictions)
    rmse = self.__rmse(ground_truths, predictions)
    return mae, mse, rmse
