import pandas as pd 
import sklearn as sk


DATA_FILE = 




data = pd.read_parquet(DATA_FILE)
model = sk.DecisionTreeRegressor(min_samples_leaf=40)
model.fit(data[TRAIN_FEATS], data[TARGET_COL])
