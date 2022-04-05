import numpy as np
import pandas as pd

# Routines required to run post prediction analysis

def move_threshhold(distance, lower_bound, upper_bound):
    '''
        Logical to define zebrafish move
    '''
    if distance != distance:
        return np.nan
    
    if (distance > lower_bound) and (distance < upper_bound):
        return 100
    else:
        return 0

def up_or_down(row):
    '''
        Logical to check  if zebrafish is in top or bottom of well
    '''
    if row['Y'] != row['Y']:
        return np.nan
    
    if row['Y'] < row['Ymid']:
        return 100
    else:
        return 0
    
def get_well_no(row, xd, yd):
    '''
        Convert (x,y) coordinate of well into single integer
    '''
    xmod = row['Xcor'] % xd
    ymod = row['Ycor'] % yd
    x = row['Xcor'] // xd
    y = row['Ycor'] // yd
    
    return ((xmod+1) + ymod*xd + y * xd * yd + x * xd * yd * 2)

def get_midpoint(p1, p2):
    '''
        Midpoint of the well
    '''
    return (p1+p2)/2

def get_orientation(RE, LE, Y):
    '''
        Get orientation of zebrafish
    '''
    midpt = get_midpoint(RE, LE)
    diff = midpt - Y
    return np.arctan2(diff[1], diff[0])*180/np.pi

def get_change_in_orientation(value):
    '''
        Get change in orientation in degrees
    '''
    if value < -180:
        return value + 360
    elif value > 180:
        return value - 360
    else:
        return value

def get_upward(row):
    """
        Check if zebrafish is facing upward
    """

    Yeyes = (row['YLE'] + row['YRE'])/2

    if (Yeyes < row['Y']):
        return 100
    else:
        return 0

def analyze_df(observations, wells, starting_image):
    '''
        Get Zebrafish behaviours from predictions

        input: 
            observations: predictions of zebrafish locations by the model
            wells: pandas dictionary of location of wells

        Output:
            observations: pandas dictionary of zebrafish behaviours
                Description of behaviours:
                    X : Location of zebrafish (X-coordinate)
                    Y : Location of zebrafish (Y-coordinate)
                    Image: Image number
                    Period: Period of stimulation
                    Well: well number
                    Xmid: midpoint of the well (X-coordinate)
                    Ymid: midpoint of the well (Y-coordinate)
                    Move: Zebrafish moved between current and previous image (Logical)
                    Up: Zebrafish is in top of well (Logical)
                    Speed: speed of the zebrafish
                    CW: orientation of zebrafish (Logical)
                    Change in orientation: change in orientation of zebrafish (degrees)
                    Edge: Check if the zebrafish on edge of well (Logical)
                    Scoot: Short movements by zebrafish (Logical)
                    Burst: Long movements by zebrafish (Logical)
                    XLE: Location of left eye (X-coordinate) 
                    YLE: Location of left eye (Y-coordinate) 
                    XRE: Location of right eye (X-coordinate)
                    YRE: Location of right eye (Y-coordinate)
                    prob_LE: Prediction probability for left eye
                    prob_RE: Prediction probability for right eye
                    prob_Y: Prediction probability for yolk
                    p_Edge: distance of zebrafish from center of well
    '''
    xd, yd = 12, 8

    radius = wells['radius'].mean()
    observations['Xmid'] = wells['radius'].mean()
    observations['Ymid'] = wells['radius'].mean()
    observations.rename(columns = {'frame':'Label'}, inplace = True)
    observations.rename(columns = {'yolk_y':'Y', 'yolk_x':'X'}, inplace = True)
    observations.rename(columns = {'right_eye_y' : 'YRE', 'right_eye_x': 'XRE', 'left_eye_y': 'YLE', 'left_eye_x': 'XLE'}, inplace = True)
    observations['Image'] = observations.index.get_level_values(0)
    observations['Label'] = observations.apply(lambda row: row['Image']+starting_image, axis = 1)
    observations['Label'] = observations.apply(lambda row: "IMG_{:04d}".format(int(row['Label'])), axis = 1)
    observations['Xcor'] = observations.index.get_level_values(1)
    observations['Ycor'] = observations.index.get_level_values(2)
    observations['MinThr'] = np.nan
    observations['MaxThr'] = np.nan
    observations['Area'] = np.nan
    observations['Up'] = observations.apply(lambda row: up_or_down(row), axis = 1)
    observations['Well'] = observations.apply(lambda row: get_well_no(row, xd, yd), axis = 1)
    observations['Exp'] = np.nan
    observations['Period'] = observations['Image'] // 100 + 1
    observations['Speed'] = np.sqrt((observations['X'] - observations['X'].shift(len(wells)))**2 + 
                                    (observations['Y'] - observations['Y'].shift(len(wells)))**2)
    observations['Move'] = observations.apply(lambda row: move_threshhold(row['Speed'], 3, np.Inf), axis = 1)
    observations['Scoot'] = observations.apply(lambda row: move_threshhold(row['Speed'], 3, 20), axis = 1)
    observations['Burst'] = observations.apply(lambda row: move_threshhold(row['Speed'], 20, np.Inf), axis = 1)
    observations['B_Up'] = np.nan
    observations['B_Up'] = observations['Up'].loc[observations['Burst'] == 100]
    observations['CW'] = (((observations['YRE'] - observations['Ymid'])**2 + (observations['XRE'] - observations['Xmid'])**2) < 
                                    ((observations['YLE'] - observations['Ymid'])**2 + (observations['XLE'] - observations['Xmid'])**2))
    observations['CW'] = observations['CW'].astype(int) * 100
    observations['Angle'] = observations.apply(lambda row: get_orientation(
                                                    RE = np.array([row['XRE'], row['YRE']]),
                                                    LE = np.array([row['XLE'], row['YLE']]),
                                                    Y = np.array([row['X'], row['Y']])
                                                    ),
                                                axis = 1)
    observations['Upw'] = observations.apply(lambda row: get_upward(row), axis = 1)
    observations['Absolute Turn'] = (observations['Angle'] - 
                                    observations['Angle'].shift(len(wells)))
    observations['Turn'] = observations.apply(lambda row: get_change_in_orientation(
                                                    row['Absolute Turn']
                                                    ), 
                                                axis = 1)
    observations['Tabs'] = observations.apply(lambda row: abs(row['Turn']), axis = 1)
    observations['Edge'] = np.sqrt((observations['Y'] - observations['Ymid'])**2 + (observations['X'] - observations['Xmid'])**2)
    observations['p_Edge'] = observations['Edge'] > 50
    observations['p_Edge'] = observations['p_Edge'].astype(int)*100
    observations.sort_values(['Image', 'Well'], inplace = True)
    observations.reset_index(drop=True, inplace=True)

    return (
        observations[[
            'Label', 
            'Area', 
            'X', 
            'Y', 
            'MinThr', 
            'MaxThr', 
            'Image', 
            'Period', 
            'Well',
            'Xmid',
            'Ymid', 
            'Exp', 
            'Move', 
            'Up',
            'Speed',
            'Scoot',
            'Burst',
            'B_Up',
            'Edge',
            'p_Edge',
            'CW',
            'Angle',
            'Upw',
            'Turn',
            'Tabs',
            'XLE',
            'YLE',
            'XRE',
            'YRE',
            'prob_LE',
            'prob_RE',
            'prob_Y'
        ]])
        
        # observations[[
        #     'Label', 
        #     'Area', 
        #     'X', 
        #     'Y', 
        #     'MinThr', 
        #     'MaxThr', 
        #     'Image', 
        #     'Period', 
        #     'Well', 
        #     'Xmid', 
        #     'Ymid', 
        #     'Exp', 
        #     'Move', 
        #     'Up', 
        #     'Speed', 
        #     'CW', 
        #     'Change in orientation',
        #     'Edge',
        #     'Scoot', 
        #     'Burst', 
        #     'XLE', 
        #     'YLE', 
        #     'XRE', 
        #     'YRE', 
        #     'prob_LE', 
        #     'prob_RE', 
        #     'prob_Y', 
        #     'p_Edge'
        # ]])