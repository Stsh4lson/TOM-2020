from modules.preprocessing import preprocess_X, preprocess_y
from starter_code.utils import load_case
import numpy as np

def trainGenerator(case_nums, batch_size):
    #takes data from begining of data set to 
    while True:
        for case_num in case_nums:
            volume, segmentation = load_case(case_num)
            #preprocessing input
            X_file = preprocess_X(volume)
            y_file = preprocess_y(segmentation)
            L = X_file.shape[0]
            batch_start = 0
            batch_end = batch_size

            while batch_start < L:
                limit = min(batch_end, L)
                X = X_file[batch_start:limit, :, :, :]
                y = y_file[batch_start:limit, :, :, :]

                #needs to yield (X, y, [None]) for some reason
                yield (X.astype(np.float32), y.astype(np.float32), [None])

                batch_start += batch_size   
                batch_end += batch_size

            if case_num == case_nums[-1]:
                # breaks loop so it starts again infinietly
                break

def validationGenerator(case_nums, batch_size):
    #generates random index from given dataset (validation data is given)
    random_val_index = case_nums[np.random.randint(0, len(case_nums))]
    volume, segmentation = load_case(random_val_index)
    #preprocessing input
    X_file = preprocess_X(volume)
    y_file = preprocess_y(segmentation)

    L = X_file.shape[0]
    batch_start = 0
    batch_end = batch_size

    while batch_start < L:
        limit = min(batch_end, L)
        X = X_file[batch_start:limit, :, :, :]
        y = y_file[batch_start:limit, :, :, :]

        #needs to yield (X, y, [None]) for some reason
        yield (X.astype(np.float32), y.astype(np.float32), [None])

        batch_start += batch_size   
        batch_end += batch_size