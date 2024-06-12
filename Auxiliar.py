import numpy as np
import math


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

    N = len(y)
    y = np.array(y)
    x = np.array(x)

    xz=np.zeros(d)
    x=np.concatenate((np.zeros(d-1),x))

    w_estimado = np.zeros( (d,N+1))
    w_estimado[:,0]=estimacion_inicial

    error = np.zeros(N)
    u = 0.1

    for i in range(N):
        #x_estimado[i-d] =
        print("-----iteracion:",i)
        print("X:",x[i:i+d])
        error[i] = y[i] - np.dot(w_estimado[:,i], x[i:i+d])
        print("Error:",error[i])
        w_estimado[:,i+1] = w_estimado[:,i] + u * x[i:i+d] * error[i]
        print("W:",w_estimado[:,i+1])


    return w_estimado



impulse_response=[0.5,1,0.2,0.1,0.05,0.01]

N = 100
p = 0.5  # Probabilidad de obtener un 1

x=generar_proceso_bernoulli(N,p)

#Calculo la respuesta del canal de comuniaciones (en esta caso el h que invente)
y=np.convolve(x,impulse_response)
d=len(impulse_response)
x=np.concatenate((x,np.zeros(d)))
d=len(impulse_response)
#print(y)
estimaciones= np.zeros((d,len(x)-d))







#x=[2,-1,0,1.5,-2,0,0.5,1,-1,0,-0.5,-1]
#y=[1.5,-1.2,-1,2,-1.5,-1,1.5,0.9,-1.3,-0.5,0,-1]
#d=3

estimaciones=LMS(x,d,y,np.zeros(d))
#
#for i in range(M-1):
#    x=generar_proceso_bernoulli(N,p)
#    y=np.convolve(x,impulse_response)
#    coeficientes[:,i+1], estimacion_salida=LMS(x,d,y,coeficientes[:,i])
#








# Fase de la respuesta en frecuencia
# Crear un gráfico

plt.figure(figsize=(10, 6))

for i in range(len(estimaciones[:,0])):
    plt.plot(estimaciones[i,:], marker='o', linestyle='-')


plt.title('Coeficientes estimados del filtro ')
plt.xlabel('Iteracion del LMS')
plt.ylabel('Valor')
plt.grid(True)




# Graficar líneas horizontales para cada valor en el vector
for i in impulse_response:
    plt.axhline(y=i, color='r', linestyle='-')

plt.ylim(0, max(impulse_response)*1.1)
plt.show()


#Ploteo la respuesta del canal y la del canal estimado con el LMS
w, h = freqz(impulse_response)
w_estim,h_estim=freqz(estimaciones[:,N])

# Magnitud de la respuesta en frecuencia
plt.figure()
plt.plot(w, 20 * np.log10(abs(h)), 'b')
plt.plot(w_estim, 20 * np.log10(abs(h_estim)), 'r')

plt.title('Respuesta en Frecuencia')
plt.xlabel('Frecuencia normalizada (rad/muestra)')
plt.ylabel('Magnitud (dB)')
plt.grid()



plt.show()
