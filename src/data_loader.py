import os
import polars as pl

train_data_path = r"..\data\raw\kaggle-jane-street-data\train.parquet"
test_data_path = r"..\data\raw\kaggle-jane-street-data\test.parquet"

def load_train_data(file_name=None):
    if file_name is None:
        return pl.read_parquet(os.path.join(train_data_path,"partition_id=0"))
    return pl.read_parquet(os.path.join(train_data_path,file_name))

def load_test_data(file_name=None):
    if file_name is None:
        return pl.read_parquet(os.path.join(test_data_path,"partition_id=0"))
    return pl.read_parquet(os.path.join(test_data_path,file_name))