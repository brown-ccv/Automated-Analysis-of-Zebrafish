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

def analyze_df(predictions, wells):

    xd, yd = 12, 8
    observations = predictions.drop(['right_eye_y', 'right_eye_x', 'left_eye_y', 'left_eye_x'],
                                axis = 1)

    radius = wells['radius'].mean()
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
    observations['Well'] = observations.apply(lambda row: get_well_no(row, xd, yd), axis = 1)
    observations['Move'] = (np.sqrt((observations['X'] - observations['X'].shift(xd * yd + 1))**2 + 
                                    (observations['Y'] - observations['Y'].shift(xd * yd + 1))**2) > 76/4)
    observations['Exp'] = np.nan
    observations['Period'] = observations['Image'] // 100 + 1
    observations['Distance'] = np.sqrt((observations['X'] - observations['X'].shift(xd * yd + 1))**2 + 
                                    (observations['Y'] - observations['Y'].shift(xd * yd + 1))**2)
    observations['Orientation'] = (((predictions['right_eye_y'] - observations['Ymid'])**2 + (predictions['right_eye_x'] - observations['Xmid'])**2) < 
                                    ((predictions['left_eye_y'] - observations['Ymid'])**2 + (predictions['left_eye_x'] - observations['Xmid'])**2))
    observations['Distance from center'] = np.sqrt((observations['Y'] - observations['Ymid'])**2 + (observations['X'] - observations['Xmid'])**2)
    observations.sort_values(['Image', 'Well'], inplace = True)
    observations.reset_index(drop=True, inplace=True)

    return observations[['Label', 'Area', 'X', 'Y', 'MinThr', 'MaxThr', 'Image', 'Period', 'Well', 'Xmid', 'Ymid', 'Exp', 'Move', 'Up', 'Distance', 'Orientation', 'Distance from center']]
