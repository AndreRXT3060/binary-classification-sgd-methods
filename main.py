from matplotlib import pyplot as plt
import numpy as np
from numba import jit, prange
import logging

import logger_config as lc
import optimization_methods as om
import function_computation as fc
import data_loader as dl


def sgd_training(
        x0: np.ndarray,
        training_images: np.ndarray,
        training_labels: np.ndarray,
        run_time: float,
        sampling_time: float,
        steplength=0.25,
        minibatch_size=8):
    
    '''Executes a timed descent using SGD;
    the last element of points contains the 
    parameters that are the result of the training'''
    
    points, times = om.sgd_minibatch(
        x0,
        training_images,
        training_labels,
        steplength,
        minibatch_size,
        run_time,
        sampling_time
    )
    
    return points, times


def stochastic_armijo_training(
        x0: np.ndarray,
        training_images: np.ndarray,
        training_labels: np.ndarray,
        run_time: float,
        sampling_time: float,
        beta=0.5,
        steplength=0.5,
        minibatch_size=8):
    
    '''Executes a timed descent using SGD with adaptive
    steplength inspired to Arjio's rule;
    the last element of points contains the 
    parameters that are the result of the training'''
    
    points, times = om.armijo_minibatch(
        x0,
        training_images,
        training_labels,
        minibatch_size,
        beta,
        steplength,
        run_time,
        sampling_time)
    
    return points, times


def sgd_momentum_training(
        x0: np.ndarray,
        training_images: np.ndarray,
        training_labels: np.ndarray,
        run_time: float,
        sampling_time: float,
        minibatch_size=8,
        eta=0.25):
    
    '''Executes a timed descent using SGD with adaptive
    steplength through momentum;
    the last element of points contains the 
    parameters that are the result of the training'''
    
    points, times = om.sgd_momentum_minibatch(
        x0,
        training_images,
        training_labels,
        minibatch_size,
        eta,
        run_time,
        sampling_time)
    
    return points, times


def deterministic_armijo_training(
        x0: np.ndarray,
        training_images: np.ndarray,
        training_labels: np.ndarray,
        run_time: float,
        sampling_time: float,
        steplength=32,
        beta=0.2,
        sigma=0.001):
    
    '''Executes a timed descent using the
    standard Armijo's rule;
    the last element of points contains the 
    parameters that are the result of the training'''
    
    points, times = om.armijo_descent(x0,
        training_images,
        training_labels,
        steplength,
        beta,
        sigma,
        run_time,
        sampling_time
    )
    
    return points, times


@jit(parallel = True)
def training_evaluation(
        test_labels: np.ndarray,
        results: np.ndarray):
    
    '''Evaluates the training results by
    classifying the images in the test set
    whose correct label is already known and
    establishing the % of correct classifications'''
    
    total_images = len(test_labels)
    errors_array = np.zeros(total_images)
    
    for i in prange(total_images):
        errors_array[i] = results[i] - test_labels[i][0]
    
    errors = np.sum(np.abs(errors_array))
    correct_classifications_ratio = (total_images - errors) / total_images
    
    return correct_classifications_ratio


def main(choice: str,
        data: dict,
        run_time: float,
        sampling_time: float):
    
    '''Executes timed descents with all the sgd variants above and the 
    traditional Armijo's rule printing the results in a graphic
    then classifies the elements of the corresponding test set to
    gauge the accuracy of the training process'''
    
    training_images = data['training_images']
    training_labels = data['training_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

    num_of_variables = len(training_images[0])
    total_images = len(training_images)
    x0 = np.random.rand(num_of_variables)
    la = 2 / total_images
    
    logger = lc.logger_setup()
    
    points0, times0 = sgd_training(x0, training_images, training_labels, run_time, sampling_time)
    f0 = [fc.objective(x, training_images, training_labels, la) for x in points0]
    print(f'SGD final loss: {f0[-1]}')
    
    points1, times1 = stochastic_armijo_training(x0, training_images, training_labels, run_time, sampling_time)
    f1 = [fc.objective(x, training_images, training_labels, la) for x in points1]
    print(f'SGD Armijo final loss: {f1[-1]}')
    
    points2, times2 = sgd_momentum_training(x0, training_images, training_labels, run_time, sampling_time)
    f2 = [fc.objective(x, training_images, training_labels, la) for x in points2]
    print(f'SGD Momentum final loss: {f2[-1]}')
    
    points3, times3 = deterministic_armijo_training(x0, training_images, training_labels, run_time, sampling_time)
    f3 = [fc.objective(x, training_images, training_labels, la) for x in points3]
    print(f'Deterministic Armijo final loss: {f3[-1]}')
    
    final_losses = {
        'SGD': f0[-1],
        'SGD Armijo': f1[-1],
        'SGD Momentum': f2[-1],
        'Deterministic Armijo': f3[-1]}
    
    try:
        plt.semilogy(times0, f0)
        plt.semilogy(times1, f1)
        plt.semilogy(times2, f2)
        plt.semilogy(times3, f3)
        plt.xlabel('time (s)')
        plt.ylabel('objective')
        plt.title('Timed descents')
        plt.legend(['SGD', 'SGD Armijo', 'SGD momentum', 'Armijo standard'])
        plt.show()
    except Exception as e:
        logger.error(f'Error, graph plot failed due to {e}')
    
    try:
        training_results = [points0[-1], points1[-1], points2[-1], points3[-1]]
        methods = ['SGD', 'SGD Armijo', 'SGD momentum', 'Armijo standard']
        for i in range(4):
            results = fc.classification(test_images, training_results[i])
            correct_classifications_ratio = training_evaluation(test_labels, results)
            print(f'{methods[i]} correctly classified {correct_classifications_ratio * 100}% of the {len(test_labels)} test images')
    except Exception as e:
        logger.error(f'Error, test images classification failed due to {e}')

    return None


if __name__ == '__main__':
    choice = '2'
    data = dl.load_data(choice)
    run_time = float(10)
    sampling_time = float(1)

    main(choice, data, run_time, sampling_time)
    
