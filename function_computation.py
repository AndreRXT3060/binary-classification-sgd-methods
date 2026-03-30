import math as m

import numpy as np
from numba import jit
from numba import prange


@jit
def logistic_regression(
        x: np.ndarray,
        image: np.ndarray,
        label: np.float64,
        total_images: int,
        la: float) -> float:
    
    '''Compute a single portion of the logistic regression
    function at the point x'''
    
    return m.log(m.exp(-label * np.dot(image, x)) + 1) + la/2 * np.dot(x, x)


@jit(parallel = True)
def objective(
        x: np.ndarray,
        images: np.ndarray,
        labels: np.ndarray,
        la: float) -> float:
    
    '''Compute the objective function at point x'''
    
    total_images = len(images)
    partial_values = np.zeros(total_images)
    
    for i in prange(total_images):
        partial_values[i] = logistic_regression(
            x,
            images[i],
            labels[i][0],
            total_images,
            la
        )
    
    return np.sum(partial_values) / total_images


@jit
def logistic_regression_gradient(
        x: np.ndarray,
        image: np.ndarray,
        label: np.float64,
        total_images: int,
        la: float) -> np.ndarray:
    
    '''Compute the gradient of a single portion of
    logistic regression at point x'''
    
    num_of_variables = len(image)
    gradient = np.zeros(num_of_variables)
    dot = np.dot(image, x)
    exp_term = np.exp(-label[0] * dot)
    
    for i in range(num_of_variables):
        gradient[i] = (
            -label[0] * image[i] * exp_term / ( exp_term + 1 )
            + la * x[i]
        )
    
    return gradient


@jit(parallel = True)
def objective_gradient(
        x: np.ndarray,
        images: np.ndarray,
        labels: np.ndarray,
        la: float) -> np.ndarray:
    
    '''Compute the gradient of the objective at point x'''
    
    total_images = len(images)
    num_of_variables = len(images[0])
    gradient = np.zeros(num_of_variables)
    subgradients = np.zeros((total_images, num_of_variables))
    
    for i in prange(total_images):
        subgradients[i] = logistic_regression_gradient(
            x, images[i], labels[i], total_images, la
        )
    
    for i in range(total_images):
        gradient  = gradient + subgradients[i]
    
    return gradient / total_images


@jit
def classification(
        images: np.ndarray,
        classification_parameters: np.ndarray) -> np.ndarray:
    
    '''Classification of a linear image using
    the training results'''
    
    total_images = len(images)
    results = np.zeros(total_images)
    

    for i in prange(total_images):
        y = np.dot(images[i], classification_parameters)
        if y > 0:
            results[i] = 1
        else:
            results[i] = -1
        
    return results
    

    



    
    
    
    