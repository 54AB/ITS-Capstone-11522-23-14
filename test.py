import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time 

fileName = 'complete_dataset.csv'
demand_dataset = pd.read_csv(fileName)

demand_dataset['school_day'] = demand_dataset['school_day'].replace({'Y':1, 'N':0})
demand_dataset['holiday'] = demand_dataset['holiday'].replace({'Y':1, 'N':0})

df = demand_dataset.copy()
df['date'] = pd.to_datetime(df['date'])
df.at[161, 'rainfall'] = 0
df.at[1377, 'rainfall'] = 0.4
df.at[1378, 'rainfall'] = 3.4
df['solar_exposure'].fillna(method = 'ffill', inplace = True)

#create day month year variable
df['year'] = df['date'].dt.year 
df['month'] = df['date'].dt.month 
df['day'] = df['date'].dt.day
df['dayOfWeek'] = df['date'].dt.dayofweek



# fig, axs = plt.subplots(2, 1, figsize=(12,12))
# fig.tight_layout(pad=5)
# ave_demand_month = df.groupby(['month', 'year'])['demand'].mean().reset_index()
# monthPlot = sns.lineplot(ave_demand_month, x="month", y="demand", hue='year', palette=sns.color_palette("husl", 6), errorbar=('ci',False), ax=axs[0])
# monthPlot.set_title('Average demand group by month and day of week', fontsize=22)
# monthPlot.set_xlabel('Month', fontsize = 20)
# monthPlot.set_xticks(range(1, 13))
# monthPlot.set_ylabel('Average Demand', fontsize=20)
# monthPlot.tick_params(labelsize = 20)
# monthPlot.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize = 20)
# monthPlot.tick_params(axis='y', labelsize = 20)
# monthPlot.legend(loc = 'upper right', fontsize=13)



# ave_demand_day = df.groupby(['dayOfWeek', 'year'])['demand'].mean().reset_index()
# print(ave_demand_day)
# dayPlot = sns.lineplot(ave_demand_day, x="dayOfWeek", y="demand", hue='year', palette=sns.color_palette("husl", 6), errorbar=('ci',False), ax=axs[1])
# # dayPlot.set_title('Average demand of each year in weekday', fontsize=15)
# dayPlot.set_xticks(range(0,7))
# dayPlot.tick_params(labelsize = 20)
# dayPlot.set_xlabel('Weekday', fontsize = 20)
# dayPlot.set_ylabel('Aveage Demand', fontsize=20)
# dayPlot.set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], fontsize = 20)
# dayPlot.tick_params(axis='y', labelsize = 20)
# dayPlot.legend(loc = 'upper right', fontsize=13)
# plt.show()

df['Max_Temp_log'] = np.log(df['max_temperature'].values)
# fig, axs = plt.subplots(1, 2, figsize=(8,5), )
# sns.histplot(x = df['max_temperature'], ax=axs[0], kde=True, color='red')
# sns.histplot(x = df['Max_Temp_log'], ax=axs[1], kde=True, color='orange')
# plt.show()

scaled_df=MinMaxScaler().fit_transform(df[['demand',
                                            'RRP',
                                            'min_temperature',
                                            #'max_temperature',
                                            'solar_exposure',
                                            'rainfall',
                                            'Max_Temp_log']])
scaled_df=pd.DataFrame(scaled_df, columns=['demand',
                                           'RRP',
                                            'min_temperature',
                                            #'max_temperature',
                                            'solar_exposure',
                                            'rainfall',
                                            'Max_Temp_log'])
scaled_df['holiday'] = df['holiday']
scaled_df['school_day'] = df['school_day']
scaled_df['month'] = df['month']
scaled_df['day'] = df['day']
scaled_df['year'] = df['year']
scaled_df['dayOfWeek'] = df['dayOfWeek']

print(scaled_df.head(n=4))
# print(scaled_df)

# heat_map, ax = plt.subplots(figsize=(10, 10)) #Set size of the heat map
# sns.heatmap(scaled_df[scaled_df.columns].corr(), annot=True, ax = ax)
# plt.title("Correlation Matrix")
# plt.show()

# train = scaled_df.loc[scaled_df.index < 1461]
# test = scaled_df.loc[scaled_df.index >= 1461]

# x_train = train.drop(labels='demand', axis=1)
# y_train = train['demand']
# x_test = test.drop(labels='demand', axis=1)
# y_test = test['demand']

# from sklearn.ensemble import RandomForestRegressor
# import time
# from sklearn.model_selection import RandomizedSearchCV

# rf_model = RandomForestRegressor(n_jobs=-1)
# param_grid = {
#     "n_estimators": np.arange(100, 1500, 100),
#     "max_depth": np.arange(6, 20, 1),
#     "max_leaf_nodes": np.arange(12, 20, 1),
# #    'max_features': ['sqrt', 'log2', None],
# }

# rdms = RandomizedSearchCV(rf_model,
#                           param_grid,
#                           scoring='neg_mean_squared_error',
#                           verbose = 2,
#                           cv=8,
#                           random_state=42,
#                           n_iter=20)
# rdms.fit(x_train, y_train)

# print(rdms.best_params_)

# rf_tuned = rdms.best_estimator_
# rf_start_time = time.time()
# rf_tuned_pred = rf_tuned.predict(x_test)
# rf_end_time = time.time()
# rf_tuned_pred = pd.Series(rf_tuned_pred, index=y_test.index)
# plt.rcParams["figure.figsize"] = (20,5)
# plt.plot(rf_tuned_pred ,label = "RF_prediction", linestyle = 'dashdot')
# plt.plot(y_test, label = 'Actual Demand')#, linestyle = 'dashdot')
# plt.legend(loc = 'upper right')
# plt.show()

# from sklearn.metrics import mean_squared_error, mean_absolute_error

# print("Execution time:", rf_end_time - rf_start_time, "seconds")
# print('Mean_squared_error:', mean_squared_error(y_test, rf_tuned_pred))
# print('Mean_absolute_error:', mean_absolute_error(y_test, rf_tuned_pred))
# print('Root mean squared error:', np.sqrt(mean_squared_error(y_test, rf_tuned_pred)))

# {'n_estimators': 1200, 'max_leaf_nodes': 19, 'max_depth': 14}
# Execution time: 0.08797478675842285 seconds
# Mean_squared_error: 0.007258249239987148
# Mean_absolute_error: 0.0687096634763948
# Root mean squared error: 0.0851953592632084
# Used all three feature


scaled_df.index = demand_dataset['date']
print(scaled_df)
pd.to_datetime(scaled_df.index)

train = scaled_df.loc[scaled_df.index < '2019-01-01']
test = scaled_df.loc[scaled_df.index >= '2019-01-01']

# train = scaled_df.loc[scaled_df.index < 1461]
# test = scaled_df.loc[scaled_df.index >= 1461]

x_train = train.drop(labels='demand', axis=1)
y_train = train['demand']
x_test = test.drop(labels='demand', axis=1)
y_test = test['demand']


from xgboost import XGBRegressor
import warnings
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import time
warnings.simplefilter(action='ignore', category=FutureWarning)
xgb = XGBRegressor(n_jobs = -1, objective = 'reg:squarederror')
xgb_params = {
        'learning_rate' : [0.001, 0.001, 0.01, 0.1, 0.2, 0.3],
        'min_child_weight': np.arange(1, 10, 1),
        "max_depth": np.arange(3, 10, 1),
        'gamma': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
        'subsample': np.arange(0.5, 1, 0.1),
        'colsample_bytree': [0.6, 0.8, 1.0],
        #'max_depth': [3, 4, 5]
}

xgb_randomSearch = RandomizedSearchCV(xgb, xgb_params, scoring='neg_mean_squared_error', verbose=1, cv=10, random_state=2, n_iter=50)
xgb_randomSearch.fit(x_train, y_train)

print(xgb_randomSearch.best_score_)
print(xgb_randomSearch.best_params_)
xgb_tuned = xgb_randomSearch.best_estimator_
xgb_start_time = time.time()
xgb_tuned_pred = xgb_tuned.predict(x_test)
xgb_end_time = time.time()
xgb_tuned_pred = pd.Series(xgb_tuned_pred, index=y_test.index)
plt.rcParams["figure.figsize"] = (20,5)
plt.plot(xgb_tuned_pred ,label = "XGB_prediction", linestyle = 'dashdot')
plt.suptitle("Prediction vs. Actual Demand - XGB", fontsize= 20)
plt.plot(y_test, label = 'Actual Demand')#, linestyle = 'dashdot')
plt.locator_params(axis='x', nbins=5)
plt.legend(loc = 'upper right', fontsize=15)
xgbPlot= plt.gca()
import matplotlib.dates as mdates
xgbPlot.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.show()
print("Execution time:", xgb_end_time - xgb_start_time, "seconds")
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print('Mean_squared_error:', mean_squared_error(y_test, xgb_tuned_pred))
print('Mean_absolute_error:', mean_absolute_error(y_test, xgb_tuned_pred))
print('Root mean squared error:', np.sqrt(mean_squared_error(y_test, xgb_tuned_pred)))
print('R^2 Score is:', r2_score(y_test, xgb_tuned_pred))


# Fitting 10 folds for each of 50 candidates, totalling 500 fits
# -0.0047626990806459045
# {'subsample': 0.5, 'min_child_weight': 2, 'max_depth': 5, 'learning_rate': 0.1, 'gamma': 0.01, 'colsample_bytree': 1.0}
# Execution time: 0.0009984970092773438 seconds
# Mean_squared_error: 0.003172404203592458
# Mean_absolute_error: 0.043073104028770924
# Root mean squared error: 0.05632409966961264
# With all created variable


# def convert_to_X_y(dataframe, window, forecasting_time):
#     dataframe = dataframe.to_numpy()
#     X = []
#     y = []
#     for i in range (window, len(dataframe) - forecasting_time + 1):
#         X.append(dataframe[i - window:i, 0: dataframe.shape[1]])
#         y.append(dataframe[i+forecasting_time - 1: i+ forecasting_time, 0])
#     return np.array(X), np.array(y)

# window_size = 7
# forecasting_time = 1
# X, y = convert_to_X_y(scaled_df, window_size, forecasting_time)


# X_train, y_train = X[:1461], y[:1461]
# X_test, y_test = X[1461:2099], y[1461:2099]

# from keras.models import Sequential
# from keras.layers import *
# from keras.losses import MeanSquaredError
# from keras.metrics import RootMeanSquaredError
# from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint
# from keras.models import load_model
# import tensorflow as tf

# tf.keras.utils.set_random_seed(1)
# tf.config.experimental.enable_op_determinism()

# lstm_model = Sequential()
# lstm_model.add(LSTM(192, activation='relu', input_shape=(X.shape[1], X.shape[2]), return_sequences= True))
# lstm_model.add(LSTM(32, activation='relu', return_sequences=False))
# lstm_model.add(Dropout(0.1))
# lstm_model.add(Dense(y.shape[1], activation = 'linear'))

# lstm_model.summary()
# cp = ModelCheckpoint('lstm_model_from_tuning/', save_best_only=True)
# lstm_model.compile(loss = 'mse',
#                     optimizer=Adam(learning_rate=0.01),
#                     metrics = [RootMeanSquaredError()])
# lstm_model.fit(X_train, y_train, epochs=50, batch_size= 10, validation_data=(X_test, y_test), verbose= 1, callbacks = [cp])
# best_LSTM = load_model('lstm_model_from_tuning/')
# lstm_start_time = time.time()
# test_prediction = best_LSTM.predict(X_test)
# lstm_end_time = time.time()
# plt.rcParams["figure.figsize"] = (20,8)
# plt.plot(test_prediction, linestyle = 'dashdot', label ='lstm_prediction')
# plt.plot(y_test, label ='Actual')
# plt.legend()
# plt.show()

# from sklearn.metrics import mean_squared_error, mean_absolute_error
# print("Execution time:", lstm_end_time - lstm_start_time, "seconds")
# print('Mean_squared_error:', mean_squared_error(y_test, test_prediction))
# print('Mean_absolute_error:', mean_absolute_error(y_test, test_prediction))
# print('Root mean squared error:', np.sqrt(mean_squared_error(y_test, test_prediction)))
# print('R^2 Score is:', r2_score(y_test, test_prediction))


# Execution time: 0.17746639251708984 seconds
# Mean_squared_error: 0.006711800736504271
# Mean_absolute_error: 0.05915758494359119
# Root mean squared error: 0.08192558047706633
# # With only day of week