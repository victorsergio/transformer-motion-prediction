import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import shapely
import math
from sklearn.preprocessing import LabelEncoder
from joblib import load
from joblib import dump
from sklearn.preprocessing import StandardScaler

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Occupancy:
    # Wrapper to be able to store an occupancy grid (numpy array nxm)
    # inside a pandas dataframe row

    def __init__(self, grid):
        self.grid = grid

    def shape(self):
        return self.grid.shape


def get_index(x_values,y_values,grid_x_values, grid_y_values):

    # - Returns an occupancy grid filled with the corresponding values
    #   from the dataset

    m = grid_y_values.shape[0]
    n = grid_x_values.shape[0]

    # create empty grid
    grid = np.zeros((m+1,n+1), dtype=int)

    # Get index for a list of points (coordinates)
    xindex = np.digitize(x_values, grid_x_values)
    yindex = np.digitize(y_values, grid_y_values)

    # fill the grid
    for (col, row) in zip(xindex, yindex):
        grid[row,col] = 1

    # Remove borders of the grid, remove outside elements, we are only interested
    # in the center of the intersection
    grid = grid[1:-1, 1:-1]
    return grid


def create_grid(location, resolution=5):

    # - Initialize an empty occupancy grid
    # :param scene: scene for the ocuppancy grid
    # :param resolution: dimension of each grid's cell (meters)
    # :return x: list of x coordinates ofeach vertex of the grid
    # :return y: list of y coordinates ofeach vertex of the grid

    location4_vertices = [(126.9483,-84.46665),
                          (170.6603,-60.82265),
                          (161.4343,-34.85665),
                          (112.8543,-54.99365)]

    location1_vertices = [(55.6866,-45.34663),
                          (66.2296,-34.218634),
                          (58.5466,-19.87363),
                          (43.6466,-31.58663)]

    location2_vertices = [(41.0835,-38.32758),
                          (65.7185,-36.00858),
                          (56.3135,-16.72358),
                          (30.7125,-21.01958)]

    location3_vertices = [(57.5857,-46.3137),
                          (65.8517,-33.4267),
                          (44.7197,-18.6437),
                          (36.0587,-29.6787)]


    if (location == 'location1'):
        center_vertices = location1_vertices
    if (location == 'location2'):
        center_vertices = location2_vertices
    if (location == 'location3'):
        center_vertices = location3_vertices
    if (location == 'location4'):
        center_vertices = location4_vertices


    CENTER = Polygon(center_vertices)

    xmin, ymin, xmax, ymax = CENTER.bounds

    # construct the rectangle of points
    x, y = np.round(np.meshgrid(np.arange(xmin, xmax, resolution), np.arange(ymin, ymax, resolution)),4)

    return x, y


def grid_labelling(df, location='location1', resolution=5):

# Calculates the occupancy grid for each frame
# param location: recording location
# param resolution: size of the grid cell in meters
# return: a labelled dataframe with occupancy grids for each frame

    # Add a new column to df to store the occupancy as object
    df = df.assign(occupancy=None)
    df['occupancy'] = df.occupancy.astype(object)

    # Create an empty grid placeholder
    x_grid, y_grid = create_grid(location,resolution=resolution)

    groups = df.groupby(['frame'])

    for name, group in groups:
        g = get_index(group.xCenter.values,group.yCenter.values,x_grid[0],np.transpose(y_grid)[0])

        # Create an Occupancy object and Store it in each frame row
        occupancy = Occupancy(g)
        df.loc[df['frame'] == name, 'occupancy'] = occupancy

    return df


def split_sequences(df, max_len = 20):

    # Split sequences to feed the neural network in training

    # Add a new column to df to store the new indexes
    df["sequenceId"] = np.nan

    current_id = 0

    groups = df.groupby(['trackId'])

    for name, group in groups:
        # how many subsequences of max length can be formed
        n_subsequences = math.ceil(len(group.index)/max_len)

        # create the indexes for the splits
        new_ids = np.arange(current_id, current_id + n_subsequences, 1)
        # create the splits (windowing)
        new_ids = np.repeat(new_ids, max_len)

        # match new ids length with index original length
        new_ids = new_ids[0:len(group.index)]

        assert len(new_ids) == len(group.index)

        # update the dataframe with the new window indexes
        serie = pd.Series(new_ids, index=group.index, name='sequenceId')
        df.update(serie)

        current_id = current_id + n_subsequences

    df['sequenceId'] = df.sequenceId.astype('int64')

    return df

def downsample(df, step=10):

    # Downsample the dataset, take a sample every 10 samples
    # original sample, 25 fps -> 80 frames = 3.2 seconds
    # downsample to 8 frames = 3.2 seconds

    df = df.groupby(['trackId'], as_index=False).apply(lambda group: group.iloc[::step]) .reset_index(drop = True, inplace = False)
    return df

def create_synthetic_id(df):
    df['globalSequenceId'] = df['recordingId'].astype('str')+"-"+df['sequenceId'].astype('str')

    le = LabelEncoder()
    df['globalSequenceId'] = le.fit_transform(df['globalSequenceId'])

    return df

def filter_columns(df):
    return df[['globalSequenceId', 'xCenter', 'yCenter', 'xCenterRelative', 'yCenterRelative', 'heading']]

def standarize_data(df, reload_scaler = False):

    scaler = StandardScaler()

    if(reload_scaler == True):
        scaler = load(data_path+'scaler.joblib')


    df_meta = df[['globalSequenceId','xCenter','yCenter']]
    df_values = df[["xCenterRelative", "yCenterRelative", "heading"]]

    scaled_values = scaler.fit_transform(df_values)
    df_values = pd.DataFrame(scaled_values, columns = df_values.columns)

    # ignore_index=False to keep column header
    df = pd.concat([df_meta, df_values], axis=1, ignore_index=False)

    if(reload_scaler == False):
        dump(scaler, data_path+'scaler.joblib')

    return df

def preprocess_file(data_path, track_number):

    # Read dataset
    tracks_csv = track_number+"_tracks.csv"
    meta_csv = track_number+"_tracksMeta.csv"

    df = pd.read_csv(data_path+tracks_csv, sep=',')
    df_meta = pd.read_csv(data_path+meta_csv, sep=',')

    # Concat meta + tracks
    df_meta = df_meta['class']
    df = df.join(df_meta, on='trackId')

    # Calculate the occupancy grids
    df = grid_labelling(df,location=recToLocation[track_number],resolution=resolution)

    # Convert to relative positions
    df['xCenterRelative'] = df.groupby(['trackId'])['xCenter'].diff().fillna(0)
    df['yCenterRelative'] = df.groupby(['trackId'])['yCenter'].diff().fillna(0)

    # downsample
    df = downsample(df, step=10)

    # Create subsequences
    df = split_sequences(df, max_len = max_len)

    # Remove sequences shorter than max_len = 20
    df = df.groupby('sequenceId').filter(lambda x: len(x) >= max_len)

    return df


def create_split(files, split_name, reload_scaler):

    data = pd.DataFrame()

    for f in files:
        df = preprocess_file(data_path, f)
        data = pd.concat([data, df], ignore_index=True)
        print(f"{f} file preprocessed ...")

    # Create a unique id for the .csv
    data = create_synthetic_id(data)
    # Select only features of interest
    data = filter_columns(data)
    # Standarize data columns for training
    data = standarize_data(data, reload_scaler = reload_scaler)

    dump(data, data_path+f"{split_name}.joblib")
    print(data_path+f"{split_name}.joblib split was created.")


# Preprocess all the files of the InD Dataset,
# calculate occupancy grids and create data splits

data_path = "./data/inD-dataset-v1.0/data/"

max_len = 20
resolution = 5

# Define recoding -> scene
recToLocation = dict.fromkeys(['00', '01', '02', '03', '04', '05', '06'], 'location4')
recToLocation.update(dict.fromkeys(['07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17'], 'location1'))
recToLocation.update(dict.fromkeys(['18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29'], 'location2'))
recToLocation.update(dict.fromkeys(['30', '31', '32'], 'location3'))

# Define data splits
train_files = ['00', '01', '02', '03', '04',
               '07', '08', '09', '10', '11', '12', '13', '14', '15',
               '18', '19', '20', '21', '22', '23', '24', '25', '26', '27',
               '30'
               ]
validation_files = ['05','16', '28', '31']
test_files = ['06','17', '29','32']

# Create train split, fit scaler
create_split(train_files, split_name="train", reload_scaler=False)
#Create validation split
create_split(validation_files, split_name="validation", reload_scaler=True)
# Create test split
create_split(test_files, split_name="test", reload_scaler=True)
