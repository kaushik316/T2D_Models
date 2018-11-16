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


# Convert each label into a one dimensional vector
def flatten(row):
    if row['is_t2d'] and row['is_lipids']:
        return 3
    elif row['is_t2d'] or row['is_lipids']:
        if row['is_t2d']:
            return 1
        else :
            return 2
    else:
        return 0


# Read in lists of chromasome ids belonging to each class
filepath = "s3://voightlab-data/"
lipids_df = pd.read_table(filepath + "lipids/lipids_groups.txt", header=None)
t2d_df = pd.read_table(filepath + "t2d/t2d_groups.txt", header=None)
both_df = pd.read_table(filepath + "grouped/Raw/lipids_T2D_overlapping_groups.txt", header=None)

# Label column
lipids_df.columns = ['snp']
t2d_df.columns = ['snp']
both_df.columns = ['snp']

grouped_df = pd.read_table(filepath + "grouped/Raw/ML_table_grouped_snpcount.txt")

# Normalize the snpcount column which is continuous, to fall between 0 and 1
if 'snpcount' in grouped_df.columns:
    grouped_df['snpcount'] = (grouped_df['snpcount'] - grouped_df['snpcount'].min())/ (grouped_df['snpcount'].max() - grouped_df['snpcount'].min())


# Remove snp since its a non numerical column - can be reattached later if we preserve order
X_train = grouped_df.loc[:, grouped_df.columns!='snp']

# Save unlabeled dataset to S3 in case we want to use it for clustering/unsupervised learning
write_to_S3(grouped_df, "s3://voightlab-data/grouped/grouped.csv")

# Label the samples based on whether they are positive for t2d, lipids or by checking for their presence 
# in the datframes segregated by type. 
grouped_df['is_t2d'] = grouped_df['snp'].apply(lambda x: 1 if t2d_df['snp'].str.contains(x).any() else 0)
grouped_df['is_lipids'] = grouped_df['snp'].apply(lambda x: 1 if lipids_df['snp'].str.contains(x).any() else 0)
grouped_df['is_both'] = grouped_df['snp'].apply(lambda x: 1 if both_df['snp'].str.contains(x).any() else 0)

#Attach a 1 dimensional label column required for certain sklearn models
grouped_df['label'] = grouped_df.loc[:, ['is_t2d', 'is_lipids']].apply(flatten, axis=1)
print (grouped_df.head())

# Remove samples/ids we want to exclude
exclude_df = pd.read_table(filepath + "grouped/Raw/ctrl_groups_to_exclude.txt", header=None)
exclude_df.columns = ['snp']
grouped_df = grouped_df[~grouped_df['snp'].isin(exclude_df['snp'])]

print ("{} samples remaining in dataset".format(len(grouped_df)))
write_to_S3(grouped_df, "s3://voightlab-data/grouped/grouped_labeled.csv")

label_cols = ['is_t2d', 'is_lipids', 'is_both']

# Select 25% of dataframe for a validation set
grouped_df = grouped_df.sample(frac=1)
idx = round(len(grouped_df) * .75)
train_df = grouped_df.iloc[:idx, :]
test_df = grouped_df.iloc[idx:, :]

# Drop the snp feature so that only numerical data remains
train_df = train_df.drop(['snp'], axis=1)
test_df = test_df.drop(['snp'], axis=1)

write_to_S3(train_df, "s3://voightlab-data/grouped/grouped_train.csv")
write_to_S3(test_df, "s3://voightlab-data/grouped/grouped_test.csv")

print ("Data saved to S3")
