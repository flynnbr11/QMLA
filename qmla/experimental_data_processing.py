import os
import numpy as np

from bisect import bisect_left
import random


def nearest_experimental_time_available(times, t):
    """
    - times: Sorted time list
    - experimental_data: dict where key is time and value is expectation value
    - t: time to get nearest available experimental data point for.

    If two times are equally close, return the smallest.
    """
    if t > max(times):
        nearest = random.choice(times)

    else:
        pos = bisect_left(times, t)
        if pos == 0:
            return times[0]
        if pos == len(times):
            return times[-1]
        before = times[pos - 1]
        after = times[pos]
        if after - t < t - before:
            nearest = after
        else:
            nearest = before

    return nearest


def nearest_experimental_expect_val_available(times, experimental_data, t):
    """
    - times: Sorted time list
    - experimental_data: dict where key is time and value is expectation value
    - t: time to get nearest available experimental data point for.

    If two times are equally close, return the smallest.
    """
    if t > max(times):
        nearest = random.choice(times)

    else:
        pos = bisect_left(times, t)
        if pos == 0:
            return times[0]
        if pos == len(times):
            return times[-1]
        before = times[pos - 1]
        after = times[pos]
        if after - t < t - before:
            nearest = after
        else:
            nearest = before

    try: 
        data_point = experimental_data[nearest]
    except: 
        print("Could not find {} in data \n{}".format(
            nearest, 
            sorted(experimental_data.keys())
        ))
        raise

    return data_point
