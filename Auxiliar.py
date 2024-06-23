import numpy as np
import math
import numpy as np
from numpy import random
from numpy.random import normal
from scipy.stats import bernoulli
import matplotlib.pyplot as plt

from matplotlib import pyplot as plt
from scipy.signal import freqz


def generar_proceso_bernoulli(N, p):
    # Genera la secuencia de bits con probabilidad p de 1 y 1-p de 0
    secuencia = np.random.choice([0, 1], size=N, p=[1-p, p])

    for i in range(len(secuencia)):
        if secuencia[i] == 0:
            secuencia[i] = -1


    return secuencia

def autocorr_(x):
    return corr_cruzada(x,x)

def corr_cruzada(x,y):
    x = np.array(x)  # Asegurarse de que x es un array de NumPy
    y = np.array(y)  # Asegurarse de que x es un array de NumPy

    N = len(x)
    autocorr_values = np.zeros(N)  # Inicializa un array de ceros

    for k in range(N):  # Itera hasta N
        autocorr_values[k] = np.sum(x[k:] * y[:N-k]) / N
    return autocorr_values

def psd(x):
    R_xx = autocorr_(x)  # Calcula la autocorrelación
    PSD = np.fft.fft(R_xx)  # Calcula la transformada de Fourier de la autocorrelación
    return np.abs(PSD)  # La PSD es el valor absoluto de la transformada de Fourier


#M es la longitud de la señal.
#K es el desplazamiento entre segmentos
def welch_psd(x, largo_ventana, paso):
    N = len(x)
    L = (N - largo_ventana) // paso + 1  # Número de segmentos con solapamiento del 50%
    window = np.hamming(largo_ventana)
    V = np.sum(abs(window)**2) / largo_ventana  # Potencia de la ventana
    psd = np.zeros(largo_ventana)

    for i in range(L):
        start = i * paso
        end = start + largo_ventana
        if end > N:
            break
        segment = x[start:end] * window
        segment_fft = np.fft.fft(segment)
        psd += np.abs(segment_fft)**2

    psd = psd / ( largo_ventana * V)  # Normalización

    return psd



#Por alguna razon te da el valor de los caeficientes de h(n) transpuestos(te da primedo el ultimo)

def LMS(x, d, y,estimacion_inicial):

    N = len(x)
    y = np.array(y)
    x = np.array(x)

    xz=np.zeros(d)
    y=np.concatenate((np.zeros(d-1),y))

    w_estimado = np.zeros( (d,N-d+1))
    w_estimado[:,0]=estimacion_inicial

    error = np.zeros(N+1)
    u = 0.05

    for i in range(N-d):
        #x_estimado[i-d] =
        #print("-----iteracion:",i)
        #print("X:",x[i:i+d])
        error[i] = y[i] - np.dot(w_estimado[:,i], x[i:i+d])
        #print("Error:",error[i])
        w_estimado[:,i+1] = w_estimado[:,i] + u * x[i:i+d] * error[i]
        #print("W:",w_estimado[:,i+1])


    return w_estimado,error



def iteracion_Jn(numero_realizaciones,largo,impulse_response,retardos,ecualizador_estimado,p,largo_ecualizador):


    u_ruido=0
    sigma_ruido=0.02
    ecualizador_LMS= np. zeros(largo_ecualizador)

    j=np.zeros((numero_realizaciones,largo+len(impulse_response)))

    for k in range(numero_realizaciones):

        #print("Largo de impulse_responde",len(impulse_response))

        #Genero una nueva entrada y una nueva repsuesta
        x_original=generar_proceso_bernoulli(largo,p)
        #Corrige el largo de x_original para que funcion el LMS(agregar la misma cantiad de 0 que el largo del canal)

        y_c=np.convolve(x_original,impulse_response)
        #Genero nuevo ruido y se lo aplico a la salida
        #print("Largo x_original",len(x_original))
        #print("Largo y_c",len(y_c))

        ruido=normal(u_ruido,sigma_ruido,len(y_c))
        #print("Largo del ruido",len(ruido))
        y_c=y_c+ruido

        d=x_original[retardos:-1]
        #Vuelvo a estimar el filtro LMS (ecualizador)
        estimaciones,error_realizacion=LMS(y_c,8,d,ecualizador_LMS)
        #print("Largo estimaciones",estimaciones.shape)
        #print("Largo de los errores",error_realizacion.shape)
        ecualizador_LMS= estimaciones[:,-1]
        #LMS deuevle todas las N iteraciones de los coeficientes del ecualizador, de todas esas me quedo con la ultima

        #print("Largo de canal_estimado",ecualizador_estimado.shape)

        #Estimo de vuelta el filtro LMS con la nueva señal con ruido
        #Vuelvo a calcular la salida del filtro LMS nuevo
        z=np.convolve(y_c, ecualizador_LMS)
        #print("Largo de x_prima", len(z))
        #print("Largo de x_prima con corchetes", len(z[len(impulse_response) + len(ecualizador_estimado) - 2:]))
        #print("Largo de x",len(x_original))

        #Guardo el ecualidor optimo (para un retardo D=8)
        if k==8:
            ecualizador_optimo=ecualizador_LMS


        #Me quedo con los ultimos N valores de x_prima, saco todos los puntos que agrega cada filtro por la convolucion
        #error=(x_prima[len(impulse_response)+len(ecualizador_estimado)-(2+corrimiento) :len(x_prima)-corrimiento]-x_original)
        #j[:,k]= (1 / numero_realizaciones) * ((abs(error) ** 2))
        j[k,:]= (1/numero_realizaciones)*abs((error_realizacion)**2)

    j_total=np.sum(j,axis=0)

    return j_total,ecualizador_optimo
