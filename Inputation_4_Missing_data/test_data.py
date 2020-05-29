import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('utils')
from datas import mill_data


data = mill_data()
train_data = data.train_data
test_data = data.test_data

print()