"""
Module to run all tasks regarding first assignment.

How you manage this module is not as important. You can choose whether you do as is written in the
skeleton or if you prefer to have input-values from the command line. However, keep the names of
all methods in the lasso-module identical as we can then correct the assignment easier.
"""

import argparse
import numpy as np 
data = 'A1_data.npy'
from lasso import lasso_ccd
from lasso import lasso_cv
from lasso import multiframe_lasso_cv
from lasso import lasso_denoise
import sounddevice as sd
import matplotlib.pyplot as plt

with open(data, "rb") as f:
    t_test = np.load(f)
    t_train = np.load(f)
    x = np.load(f)
    x_audio = np.load(f)
    x_interp = np.load(f)
    fs = np.load(f)
    n = np.load(f)
    n_interp = np.load(f)
    t = np.load(f)

def task4():
    """
    Runs code for task 4
    """
    print("Task 4")
    lamb = [0.1,1,10]
    for lam in lamb:
        w = lasso_ccd(t,x,lam)
        y = x@w
        yi = x_interp @ w
        noZ1= np.sum(np.abs(w) > 1e-10)
        noZ = np.count_nonzero(w) # Båda dessa är likvärdiga 
        print(noZ1) # När lambda ökar så sätter vi fler till noll och därav liknar får vi färre vikter. 
        print(noZ)
        

        plt.plot( n_interp, yi, '-', color='red', label='Interpolated line')  
        plt.plot(n, y, 's', label=f'Reconstructed data points λ={lam}')  #  dessa skall inte var mot n_interp 
        plt.plot(n, t, 'o', label='Original data points')  
        plt.xlabel('time')
        plt.ylabel('')
        plt.title(' LASSO reconstruction  ')
        plt.legend()
        plt.show()
        

  
def task5():
    """
    Runs code for task 5
    """
    print("Task 5")
    lambda_min = 0.01
    lambda_max = 10
    N_lambda = 10
    lambda_grid = np.exp(np.linspace(np.log(lambda_min),np.log(lambda_max), N_lambda))
    w_opt, lmbda_opt, rmse_val, rmse_est=lasso_cv(t,x, lambda_grid,5)
    plt.plot( lambda_grid, rmse_val, '-', color='blue')  
    plt.plot(lambda_grid, rmse_val, '-o',color= 'blue', label='Validation RMSE')
    plt.plot( lambda_grid, rmse_est, '-', color='red') 
    plt.plot(lambda_grid, rmse_est, '-x', color='red', label='Estimation RMSE')
    plt.axvline(lmbda_opt, linestyle='--', label=f'λ_opt={lmbda_opt:.2g}')

    plt.xlabel('λ-value')
    plt.xscale('log')
    plt.ylabel('RMSE')
    plt.title('LASSO Cross-Validation (RMSE vs λ)')
    plt.legend()
    plt.show()

    w = lasso_ccd(t,x,1.8)
    y = x@w
    yi = x_interp @ w
    plt.plot( n_interp, yi, '-', color='red', label='Interpolated line')  
    plt.plot(n, y, 's', label=f'Reconstructed data points LASSO λ={2.2}')  
    plt.plot(n, t, 'o', label='Original data points')    
    plt.xlabel('time')
    plt.ylabel('')
    plt.title('LASSO reconstruction')
    plt.legend()
    plt.show()
        
def task6():
    """
    Runs code for task 6
    """

    print("Task 6")

    lambda_min = 0.0001
    lambda_max = 1
    N_lambda = 15
    lambda_grid = np.exp(np.linspace(np.log(lambda_min),np.log(lambda_max), N_lambda))
    w_opt, lmbda_opt, rmse_val, rmse_est=multiframe_lasso_cv(t_train,x_audio, lambda_grid,5)
    plt.plot( lambda_grid, rmse_val, '-', color='blue')  
    plt.plot(lambda_grid, rmse_val, '-o',color= 'blue', label='Validation RMSE')
    plt.plot( lambda_grid, rmse_est, '-', color='red') 
    plt.plot(lambda_grid, rmse_est, '-x', color='red', label='Estimation RMSE')
    plt.axvline(lmbda_opt, linestyle='--', label=f'λ_opt={lmbda_opt:.2g}')

    plt.xlabel('λ-value')
    plt.xscale('log')
    plt.ylabel('RMSE')
    plt.title('LASSO Cross-Validation (RMSE vs λ)')
    plt.legend()
    plt.show()

def task7():
    """
    Runs code for task 7
    """
    sd.play(t_test, fs)
    sd.wait()
    denoi= lasso_denoise(t_test,x_audio,0.0052)
    sd.play(denoi, fs)
    sd.wait()
    #np.save("denoised",denoi)

    print("Task 7")


def main():
    """
    Runs a specified task given input from the user
    """

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-t",
        "--task",
        choices=["4", "5", "6", "7"],
        help="Runs code for selected task.",
    )
    args = parser.parse_args()
    try:
        if args.task is None:
            task = 0
        else:
            task = int(args.task)
    except ValueError:
        print("Select a valid task number")
        return

    if task == 4:
        task4()
    elif task == 5:
        task5()
    elif task == 6:
        task6()

    elif task == 7:
        task7()
    else:
        raise ValueError("Select a valid task number")


if __name__ == "__main__":
    main()
