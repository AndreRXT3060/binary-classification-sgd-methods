import scipy.io

import logging
import numpy as np


logger = logging.getLogger(__name__)


def load_data(choice: str):
    
    '''Load the chosen dataset between the available ones,
    choice can be either '1' or '2' '''
    
    datasets = {'1': 'datasets/MNIST_8_9.mat',
                '2': 'datasets/GISETTE.mat'}
    
    try:
        if choice == '1':
            data = scipy.io.loadmat(datasets['1'])
            images = np.transpose(np.array(data['images_8_9']))
            labels = np.array(data['labels_8_9'])
            test_images = np.transpose(np.array(data['testImages']))
            test_labels = np.array(data['testLabels_8_9'])
        elif choice == '2':
            data = scipy.io.loadmat(datasets['2'])
            images = np.ascontiguousarray(data['X_train'])
            labels = np.ascontiguousarray(data['Y_train'])
            test_images = np.ascontiguousarray(data['X_test'])
            test_labels = np.ascontiguousarray(data['Y_test'])
    except Exception as e:
        logger.error(f'Error: failed to load the dataset: {e}')
    
    if choice not in datasets:
        logger.error(f'Error, invalid dataset choice: {choice} is neither 1 or 2')
        return None
    else:
        processed_data = {
        'training_images': images,
        'training_labels': labels,
        'test_images': test_images,
        'test_labels': test_labels
        }
        return processed_data
    

