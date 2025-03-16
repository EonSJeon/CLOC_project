import sys
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import tools.file_tools


print(os.getcwd())
sample_data_path = "Our Model/data/sample_data.p"


print("Loading example data from {}".format(sample_data_path))
data = tools.file_tools.pickle_load(sample_data_path)
