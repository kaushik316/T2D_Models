import pandas as pd
import numpy as np
import boto3
import s3fs
import os
import sys
from sklearn.preprocessing import label_binarize
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from auth import access_key, secret_key


# For storage once we have labeled the dataset
def write_to_S3(df, filepath):
    bytes_to_write = df.to_csv(None).encode()
    fs = s3fs.S3FileSystem(key=access_key, secret=secret_key)
    with fs.open(filepath, 'wb') as f:
        f.write(bytes_to_write)


# Read in lists of chromasome ids belonging to each class
filepath = "s3://voightlab-data/grouped/Raw/"
BMI_df = pd.read_table(filepath + "All_BMI_groups.txt", header=None, names=['snp'])
CAD_df = pd.read_table(filepath + "All_CAD_groups.txt", header=None, names=['snp'])
T2D_df = pd.read_table(filepath + "All_T2D_groups.txt", header=None, names=['snp'])
lipids_df = pd.read_table(filepath + "All_lipid_groups.txt", header=None, names=['snp'])
grouped_df = pd.read_table(filepath + "ML_table_grouped_snpcount_normphastcon.txt")


# Label the samples based on whether they are positive for the given traits or by checking for their presence 
# in the datframes segregated by type. 
grouped_df['is_BMI'] = grouped_df['snp'].apply(lambda x: 1 if BMI_df['snp'].str.contains(x).any() else 0)
grouped_df['is_CAD'] = grouped_df['snp'].apply(lambda x: 1 if CAD_df['snp'].str.contains(x).any() else 0)
grouped_df['is_T2D'] = grouped_df['snp'].apply(lambda x: 1 if T2D_df['snp'].str.contains(x).any() else 0)
grouped_df['is_lipids'] = grouped_df['snp'].apply(lambda x: 1 if lipids_df['snp'].str.contains(x).any() else 0)


# Remove samples/ids we want to exclude
exclude_df = pd.read_table(filepath + "nonindex_ctrl_groups_to_exclude.txt", header=None, names=['snp'])
grouped_df = grouped_df[~grouped_df['snp'].isin(exclude_df['snp'])]

print ("{} samples remaining in dataset".format(len(grouped_df)))
label_cols = ['is_BMI', 'is_CAD', 'is_T2D', 'is_lipids']

# Select 25% of dataframe for a validation set
grouped_df = grouped_df.sample(frac=1)
idx = round(len(grouped_df) * .75)
train_df = grouped_df.iloc[:idx, :]
test_df = grouped_df.iloc[idx:, :]

# Drop the snp feature so that only numerical data remains
train_df = train_df.drop(['snp'], axis=1)
test_df = test_df.drop(['snp'], axis=1)

write_to_S3(train_df, "s3://voightlab-data/grouped/allgrouped_train.csv")
write_to_S3(test_df, "s3://voightlab-data/grouped/allgrouped_test.csv")

print ("Data saved to S3")