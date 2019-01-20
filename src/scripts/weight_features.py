import pandas as pd
import numpy as np
import boto3
import s3fs
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from auth import access_key, secret_key


# For storage once we have weighted the dataset
def write_to_S3(df, filepath):
    bytes_to_write = df.to_csv(None).encode()
    fs = s3fs.S3FileSystem(key=access_key, secret=secret_key)
    with fs.open(filepath, 'wb') as f:
        f.write(bytes_to_write)


# Read in lists of chromasome ids belonging to each class
filepath = "s3://voightlab-data/"

# Total number of snps
num_total = 400859.0

# Read in snps count 
weights_df = pd.read_csv(filepath + "multilabel/featcounts.csv")
grouped_traindf = pd.read_csv(filepath + "allgrouped/allgrouped_train.csv", index_col=0)

# Join training and testing df for preprocessing
grouped_testdf = pd.read_csv(filepath + "allgrouped/allgrouped_test.csv", index_col=0)
grouped_df = pd.concat([grouped_traindf, grouped_testdf])

label_cols = ['is_T2D', 'is_lipids', 'is_CAD', 'is_BMI']

# Weight each feature by the number of total snps that fall under that feature
for col in grouped_df.columns:
    if col in weights_df:
        grouped_df[col] = grouped_df[col].astype(float) * (weights_df[col].astype(float)[0] / num_total)

    else:
        if col not in label_cols:
            grouped_df[col] = grouped_df[col].astype(float) * (weights_df.iloc[0,:].mean() / num_total)


# Select 25% of dataframe for a validation set
grouped_df = grouped_df.sample(frac=1)
idx = round(len(grouped_df) * .75)
train_df = grouped_df.iloc[:idx, :]
test_df = grouped_df.iloc[idx:, :]

write_to_S3(train_df, "s3://voightlab-data/multilabel/weighted_train_df.csv")
write_to_S3(test_df, "s3://voightlab-data/multilabel/weighted_test_df.csv")

print (train_df.head())
print ("Written to S3 bucket")