import numpy as np
from dask.distributed import Client
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from models import FunkSVD

def run():
    df = pd.read_csv("data/prime_pantry_5.csv", names=["item", "user", "rating", "time"])
    df = df.drop_duplicates()
    df = df.sort_values('time').drop_duplicates(subset=['item', 'user'], keep="last")
    df = df.drop('time', axis=1)

    train = df.sample(frac=0.7, random_state=7)
    test = df.drop(train.index.tolist())

    client = Client(n_workers=2)
    model = FunkSVD(client)
    model.fit(
        n_factors=30,
        train_df=train,
        epochs=40,
        chunk_size=3000,
        collect_errors=True,
        plot_errors=True
    )

    predictions = model.predict(test)
    gt = test["rating"].to_numpy()

    eval = model.eval(gt, predictions)
    print(eval)

    client.shutdown()

    # client = Client(n_workers=2)

    # errors = []
    # evals = []

    # for i in range(10, 40, 10):
    #   model = FunkSVD(client)
    #   model.fit(
    #       n_factors=i,
    #       train_df=train,
    #       epochs=100,
    #       chunk_size=3000,
    #       collect_errors=True
    #   )

    #   errors.append(model.errors)

    #   predictions = model.predict(test)
    #   print(len(predictions))

    #   gt = test["rating"].to_numpy()
    #   print(len(gt))

    #   eval = model.eval(gt, predictions)
    #   print(eval)
    #   evals.append(eval)

    
    # mae = []
    # mse = []
    # rmse = []

    # mae1 = []
    # mse1 = []
    # rmse1 = []

    # mae2 = []
    # mse2 = []
    # rmse2 = []

    # for error in errors[0]:
    #   mae.append(error[0])
    #   mse.append(error[1])
    #   rmse.append(error[2])

    # for error in errors[1]:
    #   mae1.append(error[0])
    #   mse1.append(error[1])
    #   rmse1.append(error[2])

    # for error in errors[2]:
    #   mae2.append(error[0])
    #   mse2.append(error[1])
    #   rmse2.append(error[2])
    

    # sns.set_style("darkgrid")

    # plt.figure(0)
    # plt.ylabel('MAE')
    # plt.xlabel('Epoch')
    # plt.plot(mae)
    # plt.plot(mae1)
    # plt.plot(mae2)
    # plt.legend(["10 factors", "20 factors", "30 factors"], loc="upper right")

    # plt.savefig("res/mae.png")

    # plt.figure(1)
    
    # plt.ylabel('MSE')
    # plt.xlabel('Epoch')
    # plt.plot(mse)
    # plt.plot(mse1)
    # plt.plot(mse2)
    # plt.legend(["10 factors", "20 factors", "30 factors"], loc="upper right")
    # plt.savefig("res/mse.png")

    # plt.figure(2)
    # plt.ylabel('RMSE')
    # plt.xlabel('Epoch')
    # plt.plot(rmse)
    # plt.plot(rmse1)
    # plt.plot(rmse2)
    # plt.legend(["10 factors", "20 factors", "30 factors"], loc="upper right")
    # plt.savefig("res/rmse.png")

    # print(evals)
    

    # client.shutdown()

if __name__ == '__main__':
    run()