'''
2019 NeurIPS Submission
Title: Time-series Generative Adversarial Networks
Authors: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar

Last Updated Date: May 29th 2019
Code Author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

Data loading
(1) Load Google dataset
- Transform the raw data to preprocessed data
(2) Generate Sine dataset

Inputs
(1) Google dataset
- Raw data
- seq_length: Sequence Length
(2) Sine dataset
- No: Sample Number
- T_No: Sequence Length
- F_No: Feature Number

Outputs
- time-series preprocessed data
'''

#%% Necessary Packages
import numpy as np
import pickle


#%% Min Max Normalizer

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

#%% Load Google Data
    
def google_data_loading (seq_length):

    # Load Google Data
    x = np.loadtxt('./stock_data.csv', delimiter = ",",skiprows = 1)
    print(x.shape)

    # Flip the data to make chronological data
    x = x[::-1]
    # Min-Max Normalizer
    x = MinMaxScaler(x)
    print(type(x))
    print(x.shape)
    exit(1)
    # Build dataset
    dataX = []
    
    # Cut data by sequence length
    for i in range(0, len(x) - seq_length):
        _x = x[i:i + seq_length]
        dataX.append(_x)
        
    # Mix Data (to make it similar to i.i.d)
    idx = np.random.permutation(len(dataX))
    print(len(dataX))
    outputX = []
    for i in range(len(dataX)):
        outputX.append(dataX[idx[i]])
    print(type(outputX[0]))
    print(type(outputX))
    exit(1)
    return outputX

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

def load_solar_data(classLabel = 0):

    with open('./solar_flare.pck', 'rb') as solar_file:
        original = pickle.load(solar_file)

    upper_bound = int(original.shape[0]/4) * (classLabel + 1)
    lower_bound = upper_bound - int(original.shape[0]/4)

    original = original[lower_bound:upper_bound]
    
    modified_array = np.zeros((original.shape[0],original.shape[2], original.shape[1]-1))
    #remove the dimension 33 and check whether there are outliers -999999
    #First Tranpose everything while going through the loop
    for i in range(original.shape[0]):
        temp = np.delete(original[i],[32], axis=0)
        modified_array[i] = temp.T
    
    stacked_array = modified_array[0]
    #stack (385*60)*32 into 2D array
    for i in range(1, modified_array.shape[0]):
        stacked_array = np.concatenate((stacked_array,modified_array[i]), axis=0)
    #scale using minmax
    stacked_array = MinMaxScaler(stacked_array)
    final_array = np.zeros((modified_array.shape[0], modified_array.shape[1], modified_array.shape[2]))

    #convert back the array to 385*60*32
    final_array[0] = stacked_array[0:60]
    for i in range(1, modified_array.shape[0]):
        final_array[i] = stacked_array[i*modified_array.shape[1]: modified_array.shape[1]*(i+1)]
    
    #convert it into list 385 elements of 60*32 arrays
    lst = []
    for i in range(final_array.shape[0]):
        lst.append(final_array[i])
    
    return lst

#%% Sine Data Generation

def sine_data_generation (No, T_No, F_No):
  
    # Initialize the output
    dataX = list()

    # Generate sine data
    for i in range(No):
      
        # Initialize each time-series
        Temp = list()

        # For each feature
        for k in range(F_No):              
                          
            # Randomly drawn frequence and phase
            freq1 = np.random.uniform(0,0.1)            
            phase1 = np.random.uniform(0,0.1)
          
            # Generate Sine Signal
            Temp1 = [np.sin(freq1 * j + phase1) for j in range(T_No)] 
            Temp.append(Temp1)
        
        # Align row/column
        Temp = np.transpose(np.asarray(Temp))
        
        # Normalize to [0,1]
        Temp = (Temp + 1)*0.5
        
        dataX.append(Temp)
                
    return dataX
    
