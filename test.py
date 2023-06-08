import numpy as np
from matplotlib import pyplot as plt
import processing_signals as ps
import re

# regexp = r'-?\d+(\.\d+)?'
regexp = r'[+-]?\d+\.?\d*'
output = np.fromregex('BP1.lcn', regexp, [('num', np.float64)])

print(output['num'])

# file = open("BP1.lcn")
# test_str = file.read()
# # print(test_str)
#
#
#
# res = re.findall(r'[+-]?\d\.?\d*', test_str)

# printing result
# print(res)

