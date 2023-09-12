import numpy as np

def kernel_smooth(data, kernel, window):
    n = len(data)
    smoothed_data = np.array([np.zeros(4) for _ in range(n)])
    for i in range(n):
        numerator = np.zeros(4)
        denominator = 0
        for j in range(max(0,i-window),min(n,i+window)):
            if i == j: continue
            numerator += data[j]*kernel(data[i],data[j], 1)
            denominator += kernel(data[i],data[j], 1)
        smoothed_data[i] = numerator/denominator
    return smoothed_data

import matplotlib.pyplot as plt

# Generate some example data
x = np.linspace(0, 100, 100)
y = np.array([[1,1,1,1] for _ in range(25)])

# Define the Gaussian kernel
def gaussian_kernel(x, x_, sigma):
    return np.exp(-(x - x_)**2/(2 * sigma**2))

# Smooth the data using the Gaussian kernel
smoothed_y = kernel_smooth(y, gaussian_kernel, 3)

# Plot the original data and the smoothed data
plt.plot(x, y.flatten(), label='Original data')
plt.plot(x, smoothed_y.flatten(), label='Smoothed data')
plt.legend()
plt.savefig("fig.pdf")


##os objetivos deste projeto eram este plano de trabalho foi este
## foram feitas 2 contribuiçoes -> frame work para analise da robustez de fm
## alteraçoes a clstm

## mais informaçoes podem ser encontradas nos artigos em anexo

## Tentar fazer o artigo antes de julho 