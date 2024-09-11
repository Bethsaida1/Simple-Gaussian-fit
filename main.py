import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


energia = "pico1.txt" #nombre del archivo (recomendación: etiquetarlos con la energía o el canal al que corresponden)

guardar_data_en = "guardar_info.txt"
data = np.loadtxt(str(energia))
print(energia)

X = data[:, 0] #canales
Y = data[:, 1] #cuentas #0.232587042904406 tal vez sea necesario ajustar el número de cuentas para el background.

# Crear el histograma. 1 channel(canal) = 1 bin(barra)
counts, bin_edges = np.histogram(X, bins=len(X), weights=Y) 
bin_centers = X #Barras centradas en un número entero.

# Definir la función gaussiana+polinomio grado 1
def gaussian_plus_poly1(x, A, mu, sigma, m, b):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + m * x + b

# Calcular la incertidumbre
counts_err = np.sqrt(counts)

# Realizar el ajuste de la función gaussiana al histograma
initial_guess = [max(counts), bin_centers[np.argmax(counts)], np.std(bin_centers), 0, 0]
popt, pcov = curve_fit(gaussian_plus_poly1, bin_centers, counts, p0=initial_guess, sigma=counts_err, absolute_sigma=True) #parámetros óptimos(popt) y matriz de covarianza(pcov)

# Parámetros óptimos encontrados
A_opt, mu_opt, sigma_opt, m_opt, b_opt = popt

# Generar rejilla para dibujar la curva ajustada
X_fit = np.linspace(min(bin_centers), max(bin_centers), 1000)
Y_fit = gaussian_plus_poly1(X_fit, *popt)

# Calcular desviación estándar (incertidumbres=raiz de la diagonal de la matriz de covarianza)
perr = np.sqrt(np.diag(pcov)) #parameters_error

Verdadera_energia = (4.016979010535439e-08)*(mu_opt**2)+ 0.17159686335011792*mu_opt + 0.48508270091086614 

# Imprimir parámetros de ajuste con su incertidumbres
print(f'm = {m_opt}, b = {b_opt}')
print(f'mu = {mu_opt} ± {perr[1]}')
print(f'sigma = {round(sigma_opt,4)} ± {round(perr[2],4)}')
#print(f'm = {m_opt} ± {perr[3]}')
#print(f'b = {b_opt} ± {perr[4]}')
print(f"AREA = {round(A_opt*np.abs(sigma_opt)*np.sqrt(2*np.pi) , 4)}")
print(f"ENERGÍA = {round(Verdadera_energia, 4)}")



# Definir la función reduced_chi_squared
def reduced_chi_squared(y_obs, y_fit, y_err):
    residuals = y_obs - y_fit
    chi2 = np.sum((residuals / y_err) ** 2)
    red_chi2=chi2/(len(y_obs)-len(popt))
    return red_chi2

# Calcular e imprimir el valor de chi-cuadrado REDUCIDO para el ajuste
Y_fit_hist = gaussian_plus_poly1(bin_centers, *popt)
chi2_value = reduced_chi_squared(counts, Y_fit_hist, counts_err)
print(f"X2r = {round(chi2_value,2)}")

# Guardar parámetros en el archivo de texto
pregunta = input("¿Guardar? (s/n)")
if pregunta == "s":
    # Leer el contenido del archivo donde quiero guardar los datos
    with open(guardar_data_en, 'r') as file:
        lines = file.readlines()
    # Agregar la nueva fila
    new_row = f'{round(Verdadera_energia, 4)} {round(A_opt,4)} {round(perr[0],4)} {round(mu_opt,4)} {round(perr[1],4)} {round(np.abs(sigma_opt),4)} {round(perr[2],4)} {round(A_opt*np.abs(sigma_opt)*np.sqrt(2*np.pi) , 4)} {round(chi2_value,2)} File:{energia} \n'
    lines.append(new_row)
    with open(guardar_data_en, 'w') as file:
        file.writelines(lines)


# Graficar el histograma de los datos originales y la curva ajustada
plt.figure(figsize=(6, 3))
plt.bar(bin_centers, counts, align="center", alpha=0.7, color="blue", label="Measurement")
plt.errorbar(bin_centers, counts, yerr=counts_err, fmt="none", color="black", label="Error bars") #xerr=np.zeros(len(counts_err))
plt.plot(X_fit, Y_fit, color="red", label="Gaussian fit")
plt.xlabel("Channels")
plt.ylabel("Number of counts")
plt.yscale("log")
plt.title(energia)
plt.legend()
plt.grid(True)
plt.show()

"""
# Gráfico de residuos
residuos = (Y - Y_fit_hist)/np.sqrt(Y)
plt.figure()
plt.scatter(X, residuos, color='blue')
plt.hlines(0, min(X), max(X), colors='red', linestyles='dashed')
plt.xlabel('Canales')
plt.ylabel('Residuales')
plt.title('Análisis de residuos')
plt.grid(True)
plt.show()
"""