#!/bin/python3

import math
import os
import random
import re
import sys



#
# Complete the 'findMinimumTripsByTikRouter' function below.
#
# The function is expected to return an INTEGER.
# The function accepts FLOAT_ARRAY packet_sizes as parameter.
#

def findMinimumTripsByTikRouter(packet_sizes):
    # Write your code here
    max_parcel_size = 3.00
    packet_sizes.sort()  # in place sort
    lastoneIndex = 0

    for i, packet in enumerate(packet_sizes):
        if packet >= 2.00:
            i = i-1
            break
    lastoneIndex = i

    count = len(packet_sizes) - i-1
    start1 = 0
    end1 = lastoneIndex
    while (start1 <= end1):
        if (packet_sizes[start1] + packet_sizes[end1]) <= max_parcel_size:
            start1 += 1
            end1 -= 1
            count+=1
        else:
            count += 1
            end1 -= 1
    return count
r  = findMinimumTripsByTikRouter([1.01, 3.00, 2.5, 2.9, 1.9])
# r  = findMinimumTripsByTikRouter([1.01, 2.00, 1.99, 1.01])
print(r)