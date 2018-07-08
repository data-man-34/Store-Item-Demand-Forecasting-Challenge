import os

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))

from tqdm import tqdm

# https://www.kaggle.com/hammadkhan/xgboost-with-timeseriesfeatures
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample = pd.read_csv('../input/sample_submission.csv')
print("Train rows and columns : ", train.shape)
print("Test rows and columns : ", test.shape)

df = pd.concat([train, test])
print(df.shape)

df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['week_of_year'] = df.date.dt.weekofyear

df["median-store_item-month"] = df.groupby(['month', "item", "store"])["sales"].transform("median")
df["mean-store_item-week"] = df.groupby(['week_of_year', "item", "store"])["sales"].transform("mean")
df["item-month-mean"] = df.groupby(['month', "item"])["sales"].transform(
    "mean")  # mean sales of that item  for all stores scaled

df["store-month-mean"] = df.groupby(['month', "store"])["sales"].transform(
    "mean")  # mean sales of that store  for all items scaled

df['store_item_shifted-365'] = df.groupby(["item", "store"])['sales'].transform(
    lambda x: x.shift(365))  # sales for that 1 year  ago
df["item-week_shifted-90"] = df.groupby(['week_of_year', "item"])["sales"].transform(
    lambda x: x.shift(12).mean())  # shifted total sales for that item 12 weeks (3 months) ago

df['store_item_shifted-365'].fillna(df['store_item_shifted-365'].mode()[0], inplace=True)
df["item-week_shifted-90"].fillna(df["item-week_shifted-90"].mode()[0], inplace=True)

numeric_variables = ["median-store_item-month", "mean-store_item-week", "item-month-mean", "store-month-mean",
                     'store_item_shifted-365', "item-week_shifted-90"]

cat_variables = ["month", "week_of_year", "item", "store"]
cat_variables_ = [c + "_" for c in cat_variables]

for cat in cat_variables:
    df[cat] = df[cat].apply(lambda x: "%s_%s" % (str(x), cat))

set_cat_variables = set()

for cat in tqdm(cat_variables):
    set_cat_variables.update(set(df[cat]))

print("mapping : ")
cat_map = {value: i for i, value in enumerate(set_cat_variables)}

train = df[df.sales.notnull()]
print("new train", train.shape)
test = df[df.id.notnull()]
print("new test", test.shape)

for df in tqdm([train, test]):
    for cat in tqdm(cat_variables):
        df[cat + "_"] = df[cat].apply(lambda x: cat_map.get(x, 0))

target = "sales"

X_cat_train = np.array(train[cat_variables_])
X_cat_test = np.array(test[cat_variables_])
X_num_train = np.array(train[numeric_variables])
X_num_test = np.array(test[numeric_variables])
Y_train = train[target].values
#

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import concatenate, Flatten, Dropout, \
    GlobalAvgPool1D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam


def get_model():
    input_cat = Input((len(cat_variables),))
    input_num = Input((len(numeric_variables),))

    x_cat = Embedding(len(cat_map), 20)(input_cat)
    x_cat_1 = Flatten()(x_cat)
    x_cat_1 = Dense(20, activation="relu")(x_cat_1)
    x_cat_2 = GlobalAvgPool1D()(x_cat)

    x_num = Dense(20, activation="relu")(input_num)

    x = concatenate([x_cat_1, x_cat_2, x_num])
    x = Dropout(0.5)(x)

    x = Dense(100, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(20, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation="linear")(x)

    model = Model(inputs=[input_cat, input_num], outputs=x)
    model.compile(loss="mae", optimizer=Adam(0.01))

    model.summary()

    return model


n_bag = 5
predictions = 0
for i in range(n_bag):
    model = get_model()

    early = EarlyStopping(patience=5)
    reduce_on = ReduceLROnPlateau(patience=2)
    filepath = "baseline.h5"
    check = ModelCheckpoint(filepath)
    model.fit([X_cat_train, X_num_train], Y_train, validation_split=0.1, epochs=50,
              callbacks=[early, check, reduce_on])
    model.load_weights(filepath)
    predictions += model.predict([X_cat_test, X_num_test]) / n_bag

# In[ ]:


test[target] = predictions
test[["id", target]].to_csv("baseline.csv", index=False)
