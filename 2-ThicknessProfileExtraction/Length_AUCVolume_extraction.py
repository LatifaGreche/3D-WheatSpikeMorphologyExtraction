import os
import re
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.interpolate import splrep, BSpline
from sklearn.metrics import auc
from utils2 import *

# ----------------------------
# Robust parsers (work for result.xlsx and Data_all-spikes excel)
# ----------------------------
_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

if __name__ == "__main__":
    # Works for either file type:
    pathFile = r"C:\Users\grechel\OneDrive - Rothamsted Research\paper\github code\2-slicing\result.xlsx"
    #"...\shape_analysis_of_all_spikes_ply_Va.xlsx"

    out_xlsx = os.path.join(os.path.dirname(pathFile), "summary_spike_metrics.xlsx")
    process_file(pathFile, out_xlsx, sheet_name=0, verbose=True)
