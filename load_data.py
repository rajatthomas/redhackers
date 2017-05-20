"""
Loads data

Usage: 
    load_data.py PATH
    
Arguments:
    PATH   path to data
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from docopt import docopt

def one_hot_encode(data, encoders=None):
    """
    One hot encodes categorical variables
    
    :param data: categorical data to encoder 
    :param encoders: list of different encoder to use for encoding or None in which new encoders will be fitted
    :return: one-hot-encoded data, it's names
    """
    if encoders:
        label_encoder = encoders[0]
        onehot_encoder = encoders[1]
    else:
        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder(sparse=False)

    data = label_encoder.fit_transform(data)
    classes = label_encoder.classes_
    data = onehot_encoder.fit_transform(data[:, np.newaxis])
    encoders = [label_encoder, onehot_encoder]
    return data, classes, encoders


def calculate_weighted_by_distance(df, column_names, average_according_to='mean_elevation_m',
                                   new_feature='mean_elevation_neighbours'):

    df[new_feature] = np.nan
    coord = np.zeros((column_names.size, 2))
    elevations = np.zeros(column_names.size)

    for i, column in enumerate(column_names):
        id_province = (df.admin_L3_name == column).values
        idx_first = np.argmax(id_province)
        coord[i] = df.loc[idx_first, ['x_pos', 'y_pos']].values.astype(np.float)
        elevations[i] = df.loc[idx_first, average_according_to]

    df_coord = pd.DataFrame(data=np.column_stack((coord, elevations)), index=column_names,
                            columns=['x_pos', 'y_pos', 'elevation'])

    dist_prov = cdist(coord, coord, metric='euclidean')
    dist_prov /= dist_prov.max()

    for i, column in enumerate(column_names):
        id_province = (df.admin_L3_name == column).values
        idx_excluding_prov = np.delete(np.arange(column_names.size), i)
        dist_to_prov = dist_prov[i, idx_excluding_prov]
        elev_other = elevations[idx_excluding_prov]

        df.loc[id_province, new_feature] = np.dot(np.reciprocal(dist_to_prov), elev_other)
    return df, df_coord


def run(**kwargs):
    path_data = kwargs['PATH']

    df_data = pd.read_csv(path_data)

    # province names
    # onehot_data, classes, encoders = one_hot_encode(df_data.admin_L3_name.values)
    # df_onehot = pd.DataFrame(data=onehot_data, columns=classes)
    # del df_data['admin_L3_name']
    # df_data = pd.concat((df_data, df_onehot), axis=1)
    df_data, df_coord = calculate_weighted_by_distance(df_data, df_data.admin_L3_name.unique())

    make_plots(df_data)

    return df_data


def make_plots(df):
    pass


def plot(df):
    colors = {'Hagupit': 'r',
              'Rammasun': 'y',
              'Haiyan': 'g',
              'Melor': 'm'}
    c = df.typhoon_name.map(lambda x: colors.get(x))
    pd.scatter_matrix(df, c=c, figsize=(15, 15))
    plt.show()


if __name__ == '__main__':
    kwargs = docopt(__doc__)
    run(**kwargs)