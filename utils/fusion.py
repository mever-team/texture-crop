import numpy as np


def frequency_count(numbers, interval):
    count = 0
    for number in numbers:
        if number >= interval[0] and number < interval[1]:
            count += 1
        if number == 1 and interval[1] == 1:
            count += 1
    return count


def find_interval(number, intervals, dist):
    for intr in intervals:
        if number >= intr[0] and number < intr[1]:
            interval = intr
        if number == 1:
            interval = [1 - dist, 1]
    return interval


def weighted_average(numbers, dist=0.1):
    avg = 0
    weights = 0
    intervals = [[round(i, 1), round((i + dist), 1)] for i in np.arange(0, 1, dist)]
    for num in numbers:
        interval = find_interval(num, intervals, dist) 
        weight = frequency_count(numbers, interval) / len(numbers)
        avg += weight * num
        weights += weight 
    
    avg = avg / weights
    return avg