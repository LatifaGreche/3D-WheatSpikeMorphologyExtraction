import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, lognorm, gamma, chi2,weibull_min, beta
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score


def skewed_gaussian(x, a, loc, scale, skew):
    return a * skewnorm.pdf(x, skew, loc, scale)

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))


def lognormal(x, a, shape, loc, scale):
    return a *lognorm.pdf(x, shape, loc, scale)

def gamma_func(x,am, a, loc, scale):
    return  am*gamma.pdf(x, a,loc, scale)

def chi2_func(x, a, df, loc, scale): 
    return a*chi2.pdf(x, df, loc, scale)

def weibull_func(x, c, loc, scale):
    return weibull_min.ppf(x,c, loc, scale)

def beta_func(x,a,b,loc, scale):
    return beta.ppf((x-loc)/scale,a,b)*(1/scale)

