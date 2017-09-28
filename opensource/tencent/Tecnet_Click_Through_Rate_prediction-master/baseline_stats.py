import zipfile
import numpy as np
import pandas as pd

# load data
dfTrain = pd.read_csv("train.csv")
dfTest = pd.read_csv("test.csv")
dfAd = pd.read_csv("ad.csv")

# process data
dfTrain = pd.merge(dfTrain, dfAd, on="creativeID")
dfTest = pd.merge(dfTest, dfAd, on="creativeID")
y_train = dfTrain["label"].values

# model building
key = "appID"
dfCvr = dfTrain.groupby(key).apply(lambda df: np.mean(df["label"])).reset_index()
dfCvr.columns = [key, "avg_cvr"]
dfTest = pd.merge(dfTest, dfCvr, how="left", on=key)
dfTest["avg_cvr"].fillna(np.mean(dfTrain["label"]), inplace=True)
proba_test = dfTest["avg_cvr"].values

# submission
df = pd.DataFrame({"instanceID": dfTest["instanceID"].values, "proba": proba_test})
df.sort_values("instanceID", inplace=True)
df.to_csv("submission.csv", index=False)
with zipfile.ZipFile("submission.zip", "w") as fout:
    fout.write("submission.csv", compress_type=zipfile.ZIP_DEFLATED)