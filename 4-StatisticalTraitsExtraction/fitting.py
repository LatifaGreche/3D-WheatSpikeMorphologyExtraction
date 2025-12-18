import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, lognorm, gamma, chi2,weibull_min, beta
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
from utils5 import *

# Load the provided Excel file
file_path = r"path to ... \fitting curv\data thickness all spikes.xlsx"
path_to_axis=r"path to ... \data axis all spikes.xlsx"
data_df = pd.read_excel(file_path, header=None)
axsis_df = pd.read_excel(path_to_axis, header=None)

# Create a figure with 4x3 subplots
fig, axes = plt.subplots(3, 4, figsize=(20, 20))#(23, 5, figsize=(20, 20))
axes = axes.flatten()
output_fileo_of_fitted_curve = r"path to ... \Fitting_curves.xlsx"
output_file_combined = r"path to ... \Fitting_curves_features.xlsx"

characteristics_combined_list = []
writer= pd.ExcelWriter(output_fileo_of_fitted_curve, engine='xlsxwriter')

# Iterate over each row in the Excel file and plot on subplots
for i, (y, ax, x) in enumerate(zip(data_df.values[:12], axes, axsis_df.values[:12])):  # Process first 12 rows   ## for i, (y, ax, x) in enumerate(zip(data_df.values[:92], axes, axsis_df.values[:92])):
    # Initial guesses for curve fitting
    #y=y/data_df.max().max()
    #yy=y/max(y)
    print(i)
    initial_params_skew = [max(y), np.mean(x), np.std(x), 0.5]#[3244.40	,15.71,	62.14,	5.98]# [max(y), np.mean(x), np.std(x), 0.5]
    initial_params_gaussian = [max(y), np.mean(x), np.std(x)]#[38.14,	51.21,	31.76]# [max(y), np.mean(x), np.std(x)]
    initial_params_lognorm =[max(y), 0.1, np.min(x), np.std(x)]#[3509.34,	0.58,	-12.27,	74.27]# [max(y), 0.1, np.min(x), np.std(x)]
    initial_params_gamma = [max(y) ,1, np.mean(x), np.std(x)]#[3267.14,	2.61,	0.03,	25.70]#[max(y) ,1, np.mean(x), np.std(x)]
    initial_params_beta =[max(y) , np.mean(x), np.std(x)]#[2,5,np.mean(x), np.std(x)]#[3267.14,	2.61,	0.03,	25.70] #[max(y) , np.mean(x), np.std(x)]
    initial_params_chi2 = [1, 3, np.mean(x), np.std(x)]  # [3267.14,	5.21,	0.03,	12.85]#[1, 3, np.mean(x), np.std(x)]  #
    #6613.682942	3.483130366	11.16267759	16.38256393	98.48471183	0.869649627
#[3444.76	2.74	0.41	20.76]

    initial_params_weibull = [3244.40	, 	51.21,	31.76]
    #initial_params_beta = [2, 5,np.min(x), np.max(x)]
    # Fit the skewed Gaussian and Gaussian to the data
    try:
        params_skew, _ = curve_fit(skewed_gaussian, x, y, p0=initial_params_skew)
        params_gaussian, _ = curve_fit(gaussian, x, y, p0=initial_params_gaussian, maxfev=500000)
        params_lognorm, _ = curve_fit(lognormal, x, y, p0=initial_params_lognorm, maxfev=500000)
        params_gamma, _ = curve_fit(gamma_func, x, y, p0=initial_params_gamma, maxfev=500000)
        params_chi2, _ = curve_fit(chi2_func, x, y, p0=initial_params_chi2,  maxfev=500000)
        #params_weibull, _ = curve_fit(weibull_func, x, y, p0=initial_params_weibull, maxfev=50000)
        #params_beta, _ = curve_fit(beta_func, x/max(x), yy, p0=initial_params_beta, maxfev=500000)
        
    except RuntimeError as e:
        print(f"Fitting error on row {i + 1}: {e}")
        continue

    # Generate fitted y-values using the optimized parameters
    fittedSG_y = skewed_gaussian(x, *params_skew)
    fittedG_y = gaussian(x, *params_gaussian)
    fittedlognorm_y = lognormal(x, *params_lognorm)
    fittedGamma_y = gamma_func(x, *params_gamma)
    fittedChi2_y = chi2_func(x, *params_chi2)
    # Calculate mean, median, and mode for the skewed Gaussian
    amp_init, alpha_init, loc_init, scale_init = params_skew
    mean = skewnorm.mean(alpha_init, loc=loc_init, scale=scale_init)
    median = skewnorm.median(alpha_init, loc=loc_init, scale=scale_init)
    mode = loc_init - scale_init * alpha_init / np.sqrt(1 + alpha_init**2)

    data= pd.DataFrame({
                f'x_{i+1}': x,
                f'y_original_{i+1}': y,
                f'skewed gaussian_{i+1}':  fittedSG_y,
                f'gaussian_{i+1}': fittedG_y,
                f'lognorm_{i+1}': fittedlognorm_y,
                f'gamma_{i+1}':  fittedGamma_y,
                f'chi2_{i+1}':  fittedChi2_y
            })
    # Save the fitted curves in a new sheet (one for each row)  
    data.to_excel(writer,  sheet_name=f'ear_{i+1}',  index=False)

    mse_skew = mean_squared_error(y, fittedSG_y)
    r2_skew = r2_score(y, fittedSG_y)
    mse_gaussian = mean_squared_error(y, fittedG_y)
    r2_gaussian = r2_score(y, fittedG_y)
    mse_lognorm = mean_squared_error(y, fittedlognorm_y)
    r2_lognorm = r2_score(y, fittedlognorm_y)
    mse_gamma = mean_squared_error(y, fittedGamma_y)
    r2_gamma = r2_score(y, fittedGamma_y)
    mse_chi2 = mean_squared_error(y, fittedChi2_y)
    r2_chi2 = r2_score(y, fittedChi2_y)


    amplitude_skew, loc_skew, scale_skew, skewness_skew = params_skew
    amplitude_gaussian, mean_gaussian, stddev_gaussian = params_gaussian
    amplitude_lognorm, shape_lognorm, loc_lognorm, scale_lognorm  = params_lognorm
    amplitude_gamma, a_gamma,  loc_gamma, scale_gamma = params_gamma
    amplitude_chi2,  df_chi2, loc_chi2, scale_chi2 = params_chi2

    characteristics_combined_list.append([
    f"ear {i + 1}", amplitude_skew, loc_skew, scale_skew, skewness_skew, mse_skew, r2_skew,
    amplitude_gaussian, mean_gaussian, stddev_gaussian, mse_gaussian, r2_gaussian,
    amplitude_lognorm, shape_lognorm, loc_lognorm, scale_lognorm , mse_lognorm, r2_lognorm,
    amplitude_gamma, a_gamma,  loc_gamma, scale_gamma , mse_gamma, r2_gamma,
    amplitude_chi2,  df_chi2, loc_chi2, scale_chi2 , mse_chi2, r2_chi2
    ])
    ###################################################################################################

    ###################################################################################################
    # Plot the data and fits on the appropriate subplot
    #ax.scatter(x, y, label='Data', color='black',s=3)
    ax.plot(x, fittedSG_y, label='skewNorm', color='red',linewidth=1.0)
    ##ax.plot(x, fittedG_y, label='Gaussian', color='blue',linewidth=1.0)
    #ax.plot(x, fittedlognorm_y, label='logNormal', color='green',linewidth=1)
    ##ax.plot(x, fittedGamma_y, label='Gamma', color='orange',linewidth=1)
    #ax.plot(x, fittedChi2_y, label='Chi²', color='purple',linewidth=1)
    # Add vertical lines for mean, median, and mode
    ax.axvline(mean, color='g', linestyle='--', label=f'Mean: {mean:.2f}')
    ax.axvline(median, color='red', linestyle='--', label=f'Median: {median:.2f}')
    ax.axvline(mode, color='orange', linestyle='--', label=f'Mode: {mode:.2f}')

    #ax.plot(x, fittedweibull, label='Fitted weibull', color='red')
    #ax.plot(x, fittedbeta, label='Fitted beta', color='red')
    #ax.set_title(f"Row {i + 1}")
    #axes[6].legend(fancybox=True, framealpha=0.5, loc='best',fontsize=9) 
    if i != 0 and i != 4 and i != 8  :
            # all but last 
        ax.set_yticklabels( () )
    if i != 8 and i != 9 and i != 10  and i != 11 :
        #ax.set_yticklabels( () )
        ax.set_xticklabels( () )



writer.close()        
    # Update global y-axis limits
y_min = 0
y_max = 180
x_min = 0
x_max = 130


plt.xticks(np.arange(x_min, x_max, 20),fontsize=9)
plt.yticks(np.arange(y_min, y_max, 40), fontsize=9)
# Set the same y-axis limits for all plots
for ax in axes:
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    ax.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
# Remove remaining empty subplots (if any)
'''for ax in axes[len(data_df.values[:13]):]:
    ax.set_visible(False)'''

'''plt.xticks(np.arange(x_min, x_max, 30),fontsize=9)
plt.yticks(np.arange(y_min, y_max, 40), fontsize=9)'''

# Tight layout for better spacing
#plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.05)
plt.show()

characteristics_combined_df = pd.DataFrame(characteristics_combined_list, 
                                           columns=[
                                               "ears", "Skewed Amplitude", "Skewed Mean (loc)", "Skewed Std Dev (scale)", 
                                               "Skewed Skewness", "Skewed MSE", "Skewed R2",
                                               "Gaussian Amplitude", "Gaussian Mean", "Gaussian Std Dev", 
                                               "Gaussian MSE", "Gaussian R2",
                                                "lognorm amplitude", "lognorm shape" , "lognorm loc" , "lognorm scale" ,"lognorm mse" , "lognorm r2" ,
                                                "gamma amplitude" , "gamma a" ,  "gamma loc" , "gamma scale"  , "gamma mse" , "gamma r2" ,
                                                "chi2 amplitude" ,  "chi2 df" , "chi2 loc" , "chi2 scale"  , "chi2 mse" , "chi2 r2" 
                                    
                                           ])

# Save to Excel

characteristics_combined_df.to_excel(output_file_combined, index=False)

#print(    r2_weibull )
'''print(     characteristics_combined_df["Gaussian R2"] )
print(     characteristics_combined_df["lognorm r2"]  )
print(     characteristics_combined_df["gamma r2"]  )
print(     characteristics_combined_df["chi2 r2"]  )'''