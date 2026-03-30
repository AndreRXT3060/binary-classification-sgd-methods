import time
import random

import numpy as np
import logging
from numba import jit

import function_computation as fc


logger = logging.getLogger(__name__)


def timed_steepest_descent(
        x0: np.ndarray,
        images: np.ndarray,
        labels: np.ndarray,
        steplength: float,
        run_time: float,
        sampling_time: float):
    
    '''Classic timed steepest descent algorithm;
    x0: starting point'''
    
    x = x0.copy()
    t = float(0)
    k = 0
    
    points = [x0.copy()]
    times = [0]
    t0 = time.time()
    
    logger.info(
f'''Start steepest descent with parameters:
steplength = {steplength}
run_time = {run_time}''')
    
    try:
        while t <= run_time:
            x = x - steplength * fc.objective_gradient(x, images, labels)
            t = time.time() - t0
            if t - times[k] > sampling_time:
                points.append(x.copy())
                times.append(t)
                k = k + 1
        
        points.append(x)
        times.append(run_time)
        logger.info('Done')
        
        return points, times
    except Exception as e:
        logger.error(f'Error, steepest descent failed due to: {e}')
        raise


def armijo(
        function,
        gradient: np.ndarray,
        x: np.ndarray,
        direction: np.ndarray,
        steplength: float,
        beta: float,
        sigma: float):
    
    '''Single step of Armijo's rule;
    function: function to optimize;
    x: arguments of the function, note that
    x[0] needs to be the variable, also x is a
    tuple;
    direction: descent direction
    beta, sigma: armijo's rule parameters'''
    
    lst = list(x)
    lst[0] = x[0] + steplength * direction
    y = tuple(lst)
    fx = function(*x)
    
    while (fx - function(*y) < 
            -sigma * steplength * 
            np.dot(gradient, direction)):
        
        steplength = steplength * beta
        lst[0] = x[0] + steplength * direction
        y = tuple(lst)
    # end
    
    new_x = y[0]
    
    return new_x


def armijo_descent(
        x0: np.ndarray,
        images: np.ndarray,
        labels: np.ndarray,
        steplength: float,
        beta: float,
        sigma: float,
        run_time: float,
        sampling_time: float):
    
    '''Timed descent with armijo's rule;
    x0: starting point'''
    
    x = x0.copy()
    t = float(0)
    k = 0
    
    points = [x0.copy()]
    times = [0]
    t0 = time.time()
    total_images = len(images)
    la = 2 / total_images
    
    logger.info(
f'''Start deterministic Armijo descent with parameters:
steplength = {steplength}
reduction factor = {beta}
run_time = {run_time}''')
    
    try:
        while t <= run_time:
            gradient = fc.objective_gradient(x, images, labels, la)
            x = armijo(fc.objective,
                gradient,
                (x, images, labels, la),
                -gradient,
                steplength,
                beta,
                sigma
            )
            t = time.time() - t0
            if t - times[k] > sampling_time:
                points.append(x.copy())
                times.append(t)
                k = k + 1
    
        points.append(x)
        times.append(run_time)
        
        logger.info('Done')
        
        return points, times
    except Exception as e:
        logger.error(f'Error, armijo descent failed due to: {e}')
        raise


def sgd_minibatch(
        x0: np.ndarray,
        images: np.ndarray,
        labels: np.ndarray,
        steplength: float,
        minibatch_size: int,
        run_time: float,
        sampling_time: float):
    
    '''Timed descent with SGD minibatch'''
    
    x = x0.copy()
    t = float(0)
    k = 0
    points = [x0.copy()]
    times = [0]
    t0 = time.time()
    total_images = len(images)
    la = 2 / total_images
    
    logger.info(
f'''Start SGD descent with parameters:
steplength = {steplength}
minibatch_size = {minibatch_size}
run_time = {run_time}''')
    
    try:
        while t <= run_time:
            minibatch_indices = random.sample(
                    range(total_images),
                    minibatch_size
            )
            minibatch_images = images[minibatch_indices]
            minibatch_labels = labels[minibatch_indices]
            x = x - steplength * fc.objective_gradient(
                    x,
                    minibatch_images,
                    minibatch_labels,
                    la
            )
            t = time.time() - t0
            if t - times[k] > sampling_time:
                points.append(x.copy())
                times.append(t)
                k = k + 1
    
        points.append(x)
        times.append(run_time)
        logger.info('Done')
        
        return points, times
    except Exception as e:
        logger.error(f'Error, SGD minibatch failed due to: {e}')
        raise


@jit
def stochastic_armijo(
        x0: np.ndarray,
        minibatch_gradient: np.ndarray,
        minibatch_images: np.ndarray,
        minibatch_labels: np.ndarray,
        steplength: float,
        beta: float,
        la: float):
    
    '''Single step of a descent obtained with 
    the analogue to Armijo's rule for SGD'''
    
    c = 1 / 2
    betai = 1
    
    x = x0.copy()
    fx = fc.objective(x, minibatch_images, minibatch_labels, la)
    fy = fx
    
    ctrl_coefficient = -c * steplength * np.dot(minibatch_gradient, minibatch_gradient)
    adjusted_direction = steplength * minibatch_gradient
    
    while fy - fx > betai * ctrl_coefficient:
        betai = betai * beta
        x = x0 - betai * adjusted_direction
        fy = fc.objective(x, minibatch_images, minibatch_labels, la)
    
    return x


def armijo_minibatch(
        x0: np.ndarray,
        images: np.ndarray,
        labels: np.ndarray,
        minibatch_size: int,
        beta: float,
        steplength: float,
        run_time: float,
        sampling_time: float):
    
    '''Descent with the stochastic adaptiation
    of Armijo's rule'''
    
    x = x0.copy()
    t = float(0)
    k = 0
    points = [x0.copy()]
    times = [0]
    t0 = time.time()
    total_images = len(images)
    la = 2 / total_images
    
    logger.info(
f'''Start SGD Armijo descent with parameters:
steplength = {steplength}
reduction factor = {beta}
minibatch_size = {minibatch_size}
run_time = {run_time}''')
    
    try:
        while t <= run_time:
            minibatch_indices = random.sample(range(total_images), minibatch_size)
            minibatch_images = images[minibatch_indices]
            minibatch_labels = labels[minibatch_indices]
            minibatch_gradient = fc.objective_gradient(x, 
                minibatch_images, 
                minibatch_labels, 
                la
            )
            x = stochastic_armijo(
                x,
                minibatch_gradient,
                minibatch_images,
                minibatch_labels,
                steplength,
                beta,
                la
            )
            t = time.time() - t0
            if t - times[k] > sampling_time:
                points.append(x.copy())
                times.append(t)
                k = k + 1
    
        points.append(x)
        times.append(run_time)
        logger.info('Done')
        
        return points, times
    except Exception as e:
        logger.error(f'Error, SGD Armijo failed due to: {e}')
        raise


def sgd_momentum_minibatch(x0: np.ndarray,
        images: np.ndarray,
        labels: np.ndarray,
        minibatch_size: int,
        eta: float,
        run_time: float,
        sampling_time: float):
    
    '''Descent with SGD combined with momentum based
    steplength adaptation'''
    
    x = x0.copy()
    t = float(0)
    k = 0
    iteration = 0
    points = [x0.copy()]
    times = [0]
    t0 = time.time()
    total_images = len(images)
    la = 2 / total_images
    num_of_variables = len(images[0])
    momentum = np.zeros(num_of_variables)
    logger.info(
f'''Start SGD momentum descent with parameters:
steplength = {eta}
minibatch_size = {minibatch_size}
run_time = {run_time}''')
    
    try:
        while t <= run_time:
            minibatch_indices = random.sample(range(total_images), minibatch_size)
            minibatch_images = images[minibatch_indices]
            minibatch_labels = labels[minibatch_indices]
            minibatch_gradient = fc.objective_gradient(x, minibatch_images, minibatch_labels, la)
            steplength = 2 * eta / (iteration + 3)
            beta = iteration / (iteration + 2)
            momentum = beta*momentum + minibatch_gradient
            x = x - steplength * momentum
            iteration = iteration + 1
            t = time.time() - t0
            if t - times[k] > sampling_time:
                points.append(x.copy())
                times.append(t)
                k = k + 1
    
        points.append(x.copy())
        times.append(run_time)
        
        logger.info('Done')
        
        return points, times
    except Exception as e:
        logger.error(f'Error, SGD momentum failed due to: {e}')
        raise
        
