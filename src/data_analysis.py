import pandas as pd
import numpy as np

def up_or_down(row):
    
    if row['Y'] < row['Ymid']:
        return 100
    else:
        return 0
    
def get_well_no(row, xd, yd):

    xmod = row['Xcor'] % xd
    ymod = row['Ycor'] % yd
    x = row['Xcor'] // xd
    y = row['Ycor'] // yd
    
    return ((xmod+1) + ymod*xd + y * xd * yd + x * xd * yd * 2)

def analyze_df(observations, wells):

    xd, yd = 12, 8

    radius = wells['radius'].mean()
    observations['Label'] = np.nan
    observations['Xmid'] = wells['radius'].mean()
    observations['Ymid'] = wells['radius'].mean()
    observations.rename(columns = {'yolk_y':'Y', 'yolk_x':'X'}, inplace = True)
    observations.rename(columns = {'right_eye_y' : 'YRE', 'right_eye_x': 'XRE', 'left_eye_y': 'YLE', 'left_eye_x': 'XLE'}, inplace = True)
    observations['Image'] = observations.index.get_level_values(0)
    observations['Xcor'] = observations.index.get_level_values(1)
    observations['Ycor'] = observations.index.get_level_values(2)
    observations['MinThr'] = np.nan
    observations['MaxThr'] = np.nan
    observations['Area'] = np.nan
    observations['Up'] = observations.apply(lambda row: up_or_down(row), axis = 1)
    observations['Well'] = observations.apply(lambda row: get_well_no(row, xd, yd), axis = 1)
    observations['Move'] = (np.sqrt((observations['X'] - observations['X'].shift(len(wells)))**2 + 
                                    (observations['Y'] - observations['Y'].shift(len(wells)))**2) > 76/4)
    observations['Move'] = observations['Move'].astype(int)*100
    observations['Exp'] = np.nan
    observations['Period'] = observations['Image'] // 100 + 1
    observations['Speed'] = np.sqrt((observations['X'] - observations['X'].shift(len(wells)))**2 + 
                                    (observations['Y'] - observations['Y'].shift(len(wells)))**2)
    observations['CW'] = (((observations['YRE'] - observations['Ymid'])**2 + (observations['XRE'] - observations['Xmid'])**2) < 
                                    ((observations['YLE'] - observations['Ymid'])**2 + (observations['XLE'] - observations['Xmid'])**2))
    observations['CW'] = observations['CW'].astype(int) * 100
    observations['Edge'] = np.sqrt((observations['Y'] - observations['Ymid'])**2 + (observations['X'] - observations['Xmid'])**2)
    observations.sort_values(['Image', 'Well'], inplace = True)
    observations.reset_index(drop=True, inplace=True)

    return observations[['Label', 'Area', 'X', 'Y', 'MinThr', 'MaxThr', 'Image', 'Period', 'Well', 'Xmid', 'Ymid', 'Exp', 'Move', 'Up', 'Speed', 'CW', 'Edge', 'XLE', 'YLE', 'XRE', 'YRE']]
