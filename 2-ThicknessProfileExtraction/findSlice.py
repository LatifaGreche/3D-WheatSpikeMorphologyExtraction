import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
def double_logi(x, a, b, c, d, e, f):
    y = a + (b / (1 + (np.exp(-1 * ((x - c + (d / 2)) / e))))) * (
        1 - (1 / (1 + np.exp(-1 * ((x - c - (d / 2)) / f)))))
    return y

def find_slice(y):       
    guess_logi = np.array([0.4, 0.4, 77, 105.8, 3.8, 7])
    x = np.arange(0, len(y))
    poptt, _ = opt.curve_fit(double_logi, x, y, guess_logi, method="trf", maxfev=1000000)
    y_fit_logi_double = double_logi(x, *poptt)
    x2 = np.where(y_fit_logi_double==max(y_fit_logi_double))[0][0]
    y2 = y_fit_logi_double[x2]
    first_grad = np.gradient(y_fit_logi_double)
    second_grad = np.gradient(first_grad)
    infls = np.where(np.diff(np.sign(second_grad)))[0]
    x1 = infls[0]
    y1 = y_fit_logi_double[x1]
    x3 = infls[1]
    y3 = y_fit_logi_double[x3]
    X = [x1, x2]
    Y = [y1, y2]
    X2 = [x2, x3]
    Y2 = [y2, y3]
    polynomial1 = np.poly1d(np.polyfit(X, Y, 1))
    polynomial2 = np.poly1d(np.polyfit(X2, Y2, 1))
    first_ = x[x1:x2+1]
    second_ = x[x2:x3+1]
    y_axis1 = polynomial1(first_)
    y_axis2 = polynomial2(second_)
    diff_first = []
    for i in range(0, len(first_)):
        diff = y_fit_logi_double[first_[i]]-y_axis1[i]
        diff_first.append(diff)
    idx_first_curve = np.argmax(diff_first)
    x_point_first = first_[idx_first_curve]
    y_point_first = y_fit_logi_double[x_point_first]
    diff_second = []
    for i in range(0, len(second_)):
        diff = y_fit_logi_double[second_[i]]-y_axis2[i]
        diff_second.append(diff)
    idx_second_curve = np.argmax(diff_second)
    x_point_second = second_[idx_second_curve]
    y_point_second = y_fit_logi_double[x_point_second]
    first_point = (x_point_first, y_point_first)
    second_point = (x_point_second, y_point_second)
    #print(first_point, second_point)
    return first_point[0] ,second_point[0]