import numpy as np
import pandas as pd
from math import ceil
import warnings
warnings.filterwarnings('ignore')
import joblib
list_of_cols = joblib.load('../data/cols_pattern.pkl')

def week_of_month(dt):
    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()
    return int(ceil(adjusted_dom/7.0))


features_data = pd.read_csv('../data/Features_data_set.csv')
sales_data = pd.read_csv('../data/sales_data_set.csv')
stores_data = pd.read_csv('../data/stores_data_set.csv')


df = pd.merge(sales_data, features_data, on=['Date', 'Store'], how = 'left')
if all(df.IsHoliday_x.values == df.IsHoliday_y.values):
        df.drop('IsHoliday_y', axis=1, inplace=True)
        df.rename(columns={'IsHoliday_x': 'IsHoliday'}, inplace=True)

df = pd.merge(df, stores_data, on='Store', how='left')
df.Date = pd.to_datetime(df.Date, format = '%d/%m/%Y')
ori_df = df.copy()

 



def create_weekly_sales(df):
    #I will cut off outlier values at 98% quantile as distribution for lgbm doesnt have to be normal I wont logarithmize the target
    df['year'] = df.Date.dt.year
    df['month'] = df.Date.dt.month
    df['week_of_year'] = df.Date.dt.weekofyear
    df['week_of_month'] = df.Date.map(week_of_month)
    date_features = list(df.columns[-4:])
    df['Weekly_Sales'].loc[(df.Weekly_Sales > np.quantile(df.Weekly_Sales, 0.98))&(df.year != 2012)] = np.quantile(df.Weekly_Sales, 0.98)
    #only in train set (wo 2012)
    df = df[date_features + ['Date','Store', 'Dept', 'Weekly_Sales']]
    ddf = pd.get_dummies(df.drop(['Date', 'Weekly_Sales'], axis = 1), columns = date_features+['Store', 'Dept'], dummy_na=True)
    ddf['year_2013'] = 0
    if not 'week_of_month_6' in ddf.columns:
        ddf['week_of_month_6'] = 0
    ddf.columns = [x.replace('.0','').lower() for x in list(ddf.columns)]
    ddf = ddf.reindex(sorted(ddf.columns), axis=1)
    dummy_df = pd.concat([ddf,df['Weekly_Sales']], axis = 1)
    dummy_df.to_csv('../data/dummy_weekly_sales.csv', index = False)
    print('weekly sales dataset saved')




def create_total_sales(df):
    df = df.groupby(['Date']).Weekly_Sales.sum().reset_index().rename(columns = {'Weekly_Sales':'total_sales'})
    df['year'] = df.Date.dt.year
    df['month'] = df.Date.dt.month
    df['week_of_year'] = df.Date.dt.weekofyear
    df['week_of_month'] = df.Date.map(week_of_month)
    df['total_sales'].loc[(df.total_sales > np.quantile(df.total_sales, 0.98))&(df.year != 2012)] = np.quantile(df.total_sales, 0.98)
    date_features = list(df.columns[-4:])
    ddf = pd.get_dummies(df.drop(['Date', 'total_sales'], axis = 1), columns = date_features, dummy_na=True)
    ddf['year_2013'] = 0
    ddf.columns = [x.replace('.0','').lower() for x in list(ddf.columns)]
    if not 'week_of_month_6' in ddf.columns:
        ddf['week_of_month_6'] = 0
    ddf = ddf.reindex(sorted(ddf.columns), axis=1)
    dummy_df = pd.concat([ddf,df['total_sales']], axis = 1)
    
    dummy_df.to_csv('../data/dummy_total_sales.csv', index = False)
    print('total sales dataset saved')



def create_markdowns(df):
    df = df[['Store','Dept','Date', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']]
    df['year'] = df.Date.dt.year
    df['month'] = df.Date.dt.month
    df['week_of_year'] = df.Date.dt.weekofyear
    df['week_of_month'] = df.Date.map(week_of_month)
    df = df.dropna()
    date_features = list(df.columns[-4:])
    markdowns = ['MarkDown1','MarkDown2','MarkDown3', 'MarkDown4','MarkDown5']
    ddf = pd.get_dummies(df.drop(['Date']+markdowns, axis = 1), columns = date_features+['Store', 'Dept'], dummy_na=True)
    ddf['year_2013'] = 0
    ddf.columns = [x.replace('.0','').lower() for x in list(ddf.columns)]
    ddf[list(set(list_of_cols) - set(ddf.columns))] = 0
    ddf = ddf.reindex(sorted(ddf.columns), axis=1)
    dummy_df = pd.concat([ddf,df[markdowns]], axis = 1)
    dummy_df.to_csv('../data/dummy_markdowns.csv', index = False)
    print('markdowns dataset saved')

create_weekly_sales(df)
create_total_sales(df)
create_markdowns(df)