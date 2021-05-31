import pandas as pd
import json

if __name__ == '__main__':
    with open("./data/Prime_Pantry_5.json", "r") as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]

    df = pd.DataFrame(data)
    df = df[["reviewerID", "asin", "overall", "unixReviewTime"]]
    df.columns = ["user", "item", "rating", "time"]
    df.to_csv("./data/prime_pantry_5.csv", index=False, header=False)