import pandas as pd
import numpy as np
import boto3
import s3fs
import os
import sys
from sklearn.model_selection import train_test_split
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from auth import access_key, secret_key


# Read in data and compare class balances
def show_counts(df, pos_col, neg_col):
    positives = df['type'].value_counts()[pos_col]
    controls = df['type'].value_counts()[neg_col]
    total = len(df)
    print ("Total samples: {} \nPositives: {} \nControls: {}".format(total, positives, controls))


# For storage 
def write_to_S3(df, filename):
    bytes_to_write = df.to_csv(None).encode()
    fs = s3fs.S3FileSystem(key=access_key, secret=secret_key)
    with fs.open(filename, 'wb') as f:
        f.write(bytes_to_write)


# Alter with lipids/t2d/other if needed
options = ["t2d", "lipids"]

for option in options:
    trait = option

    train_filepath = "S3://voightlab-data/{}/{}_training_ML_table.txt".format(trait, trait)
    test_filepath = "S3://voightlab-data/{}/{}_testing_ML_table.txt".format(trait, trait)

    # Used to name the output files
    out_train = "voightlab-data/{}/{}_train.csv".format(trait, trait)
    out_test = "voightlab-data/{}/{}_test.csv".format(trait, trait)

    train_df = pd.read_table(train_filepath)
    test_df = pd.read_table(test_filepath)
    train_df.tail()


    # Change labels to 0 or 1 for control or positive
    train_df['type'] = train_df['type'].apply(lambda x: 0 if x == 'control' else 1)
    test_df['type'] = test_df['type'].apply(lambda x: 0 if x == 'control' else 1)


    # Drop the snp feature so that only numerical data remains
    train_df = train_df.drop(['snp'], axis=1)
    test_df = test_df.drop(['snp'], axis=1)

    # Write out data to file so we don't have to repeat preprocessing 
    # every time we train a model from a different file
    write_to_S3(train_df, out_train)
    write_to_S3(test_df, out_test)

print ("Data saved to S3")


