import pickle
import numpy as np

def delete_outlier(data):
    
    lst_idx = []
    counter = 0
    i = 0
    for el in data:
        if -99999. in el:
            counter= counter+1
            lst_idx.append(i)
        i=i+1
    print(lst_idx)
    X = np.delete(data, lst_idx, axis=0)

    return X

def process_solar_data_professors_approach(classLabel = 0):

    with open('./solar_flare.pck', 'rb') as solar_file:
        original = pickle.load(solar_file)

    upper_bound = int(original.shape[0]/4) * (classLabel + 1)
    lower_bound = upper_bound - int(original.shape[0]/4)

    ori_data = delete_outlier(original[lower_bound:upper_bound])

    return ori_data

d = process_solar_data_professors_approach()

