import pandas as pd
import numpy as np

def up_or_down(row):
    
    if row['Y'] > row['Ymid']:
        return 100
    else:
        return 0
    
def get_well_no(row):
    
    xd = 

    xmod = row['Xcor'] % 12
    ymod = row['Ycor'] % 8
    x = row['Xcor'] // 12
    y = row['Ycor'] // 8
    
    return ((xmod+1) + ymod*12 + x * 12 * 8 + y * 12 * 8 * 2)

def get_period(row):
    
    return row['Image']//100 + 1

def analyze_df(predictions, wells):

    observations = predictions.drop(['right_eye_y', 'right_eye_x', 'left_eye_y', 'left_eye_x'],
                                axis = 1)
    observations['Label'] = np.nan
    observations['Xmid'] = wells['radius'].mean()
    observations['Ymid'] = wells['radius'].mean()
    observations.rename(columns = {'yolk_y':'Y', 'yolk_x':'X'}, inplace = True)
    observations['Image'] = observations.index.get_level_values(0)
    observations['Xcor'] = observations.index.get_level_values(1)
    observations['Ycor'] = observations.index.get_level_values(2)
    observations['MinThr'] = np.nan
    observations['MaxThr'] = np.nan
    observations['Area'] = np.nan
    observations['Up'] = observations.apply(lambda row: up_or_down(row), axis = 1)
    observations['Well'] = observations.apply(lambda row: get_well_no(row), axis = 1)
    observations['Move'] = (np.sqrt((observations['X'] - observations['X'].shift(len(wells) + 1))**2 + 
                                    (observations['Y'] - observations['Y'].shift(len(wells) + 1))**2) > 76/4)
    observations['Exp'] = np.nan
    observations['Period'] = observations.apply(lambda row: get_period(row), axis = 1)
    observations.sort_values(['Image', 'Well'], inplace = True)
    observations.reset_index(drop=True, inplace=True)

    return observations[['Label', 'Area', 'X', 'Y', 'MinThr', 'MaxThr', 'Image', 'Period', 'Well', 'Xmid', 'Ymid', 'Exp', 'Move', 'Up']]
