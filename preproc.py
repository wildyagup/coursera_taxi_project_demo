
# coding: utf-8

# In[8]:


import pandas as pd
import pickle
import numpy as np
import geopandas as gpd
from shapely.geometry.polygon import Polygon
import matplotlib.cm as cm
import matplotlib.colors as clrs

def preprocess_data():
    '''
    Load may and june predicted and actual trip counts and bind geo data.
    '''
    
    ### load predictions
    preds_May = pd.read_csv('./data/preds_may.csv', parse_dates=['time'])
    preds_June = pd.read_csv('./data/preds_june.csv', parse_dates=['time'])

    ### preprocess may predictions
    pr_May = preds_May[['time', 'region', '1']].copy()
    pr_May.loc[:, 'time'] = pr_May.loc[:, 'time'] + pd.DateOffset(hours=1)

    may_last_hours = pd.date_range(pr_May.time.iloc[-1]+pd.DateOffset(hours=1),
                                   pr_May.time.iloc[-1]+pd.DateOffset(hours=5),freq='h')

    last_may_counts = []
    for i in range(len((may_last_hours))):
        last_may_counts.append(preds_May[preds_May.time == preds_May.time.iloc[-1]].loc[:, str(i+2)].values)

    may_last = pd.DataFrame({'time':np.array([[t]*preds_May.region.nunique() for t in may_last_hours]).ravel(),
                             'region':np.array([preds_May[preds_May.time == preds_May.time.iloc[-1]].region]*5).ravel(),
                             '1': np.array(last_may_counts).ravel()},
                             columns=['time','region','1'])

    df_May = pd.concat([pr_May, may_last], ignore_index=True).rename(columns={'1': 'pred_counts'}, index=str)

    ### preprocess june predictions
    pr_June = preds_June[['time', 'region', '1']].copy()
    pr_June.loc[:, 'time'] = pr_June.loc[:, 'time'] + pd.DateOffset(hours=1)

    june_last_hours = pd.date_range(pr_June.time.iloc[-1]+pd.DateOffset(hours=1),
                                   pr_June.time.iloc[-1]+pd.DateOffset(hours=5),freq='h')

    last_june_counts = []
    for i in range(len((june_last_hours))):
        last_june_counts.append(preds_June[preds_June.time == preds_June.time.iloc[-1]].loc[:, str(i+2)].values)

    june_last = pd.DataFrame({'time':np.array([[t]*preds_June.region.nunique() for t in june_last_hours]).ravel(),
                             'region':np.array([preds_June[preds_June.time == preds_June.time.iloc[-1]].region]*5).ravel(),
                             '1': np.array(last_june_counts).ravel()},
                             columns=['time','region','1'])

    df_June = pd.concat([pr_June, june_last], ignore_index=True).rename(columns={'1': 'pred_counts'}, index=str)

    ### combine may and june
    preds_df = pd.concat([df_May, df_June], ignore_index=True)

    ### load actual data
    dfs = []
    periods = ['2016-0'+str(i) for i in range(5,7)]

    for period in periods:
        with open('./data/' + period + '_agg.csv') as file:
            df = pd.read_csv(file, parse_dates=['time'])
            dfs.append(df)

    resdf = pd.concat(dfs, ignore_index=True)
    resdf = resdf[resdf.region.isin(df_May.region.unique())]

    ### combine predictions and actual
    df = pd.merge(resdf, preds_df, on=['time', 'region'])
    regions = df.region.unique()

    ### add geo data
    reg_data = pd.read_csv('./data/regions.csv', sep=';')
    reg_data['geometry'] = reg_data.apply(lambda row: Polygon([
        (row['west'], row['north']),
        (row['east'], row['north']),
        (row['east'], row['south']),
        (row['west'], row['south'])
    ]), axis=1)

    df_geo = gpd.GeoDataFrame(pd.merge(df, reg_data[['region','geometry']], on='region', how='left'))

    return df_geo

def create_styledicts(df_geo):
    '''
    Create styledicts for folium TimeSliderChoropleth control.
    '''
    
    ### styledict for predicted data
    stdict_pred = {
        str(reg): {
            ts.to_pydatetime(): {
                'color': clrs.to_hex(cm.viridis_r(np.int(df_geo[(df_geo.region==reg)&(df_geo.time==ts)].pred_counts))),
                'opacity': 0.5
            } for ts in df_geo[df_geo.region==reg].time   
        } for reg in df_geo[df_geo.time == df_geo.time.iloc[0]][['region', 'geometry']].set_index('region').index
    }
    
    ### styledict for actual data
    stdict_fact = {
        str(reg): {
            ts.to_pydatetime(): {
                'color': clrs.to_hex(cm.viridis_r(np.int(df_geo[(df_geo.region==reg)&(df_geo.time==ts)].counts))),
                'opacity': 0.5
            } for ts in df_geo[df_geo.region==reg].time   
        } for reg in df_geo[df_geo.time == df_geo.time.iloc[0]][['region', 'geometry']].set_index('region').index
    }
    
    ### dump styledicts to disk
    with open('./data/stdict_pred.pkl', 'wb') as f:
        pickle.dump(stdict_pred, f)
        
    with open('./data/stdict_fact.pkl', 'wb') as f:
        pickle.dump(stdict_fact, f)