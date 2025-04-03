"""
Here we implement a quickelect algorithm to randomly select data from a dataset to create a random subset - this is our placebo group
"""

import random


def quickselect(arr, k):
    if len(arr) == 1:
        return arr[0]
    pivot = random.choice(arr)
    lows = [x for x in arr if x < pivot]
    highs = [x for x in arr if x > pivot]
    pivots = [x for x in arr if x == pivot]

    if k < len(lows):
        return quickselect(
            lows, k
        )  # this case means that the kth smallest number in arr is in the lows array
    elif k < len(lows) + len(pivots):
        return pivots[
            0
        ]  # this case means that the kth smallest number in arr is in the pivots array
    else:
        return quickselect(
            highs, k - len(lows) - len(pivots)
        )  # this case means that the kth smallest number in arr is in the highs array and thus would be the kth - len(lows) - len(pivots)th smallest number in highs
