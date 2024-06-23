import numpy as np
from numpy import random
from numpy.random import normal
from scipy.signal import freqz
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import random

from Auxiliar import autocorr_, psd, welch_psd, corr_cruzada, LMS, iteracion_Jn


def generar_proceso_bernoulli(N, p):
    # Genera la secuencia de bits con probabilidad p de 1 y 1-p de 0
    secuencia = np.random.choice([0, 1], size=N, p=[1-p, p])

    for i in range(len(secuencia)):
        if secuencia[i] == 0:
            secuencia[i] = -1


    return secuencia





# Define la longitud del proceso Bernoulli y la probabilidad p
N = 200
p = 0.5  # Probabilidad de obtener un 1


x=generar_proceso_bernoulli(N,p)

plt.figure(figsize=(10, 4))
plt.plot(x, linestyle='None')  # Gráfica de puntos
plt.title('Proceso de Bernoulli ')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.ylim(-1.5, 1.5)  # Ajuste de los límites del eje y
plt.grid(True)
plt.show()


#Declaro un modelo del canal en tiempo discreto

h=[0.5,1,0.2,0.1,0.05,0.01]
u=0
sigma=0.002
ruido=normal(u,sigma,N)

plt.figure(figsize=(10, 4))
plt.plot(ruido, marker='o', linestyle='None')  # Gráfica de puntos
plt.title('Muestras del ruido ')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.grid(True)
plt.show()


y=np.convolve(x,h)
y = y[:-(len(h)-1)] #Elimino los ultimos elementos para que quede del mismo largo de x



plt.figure(figsize=(10, 4))
plt.plot(y, marker='o', linestyle='None')  # Gráfica de puntos
plt.title('Respuesta del canal(discreto) a x')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.grid(True)
plt.show()



Ry = autocorr_(y)
Rx=autocorr_(x)


psd_y=welch_psd(Ry,30,15)
psd_x=welch_psd(Rx,len(x),25)




plt.figure(figsize=(10, 4))
plt.plot(Rx, marker='o', linestyle='-')  # Gráfica de puntos
plt.title('Correlacion X')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(psd_x, marker='o', linestyle='-')  # Gráfica de puntos
plt.title('PSD X')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(Ry, marker='o', linestyle='-')  # Gráfica de puntos
plt.title('Correlacion Y')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(psd_y, marker='o', linestyle='-')  # Gráfica de puntos
plt.title('PSD Y')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.grid(True)
plt.show()



##Ejericio2

Ry=autocorr_(y)
Ryx= corr_cruzada(y,x)






impulse_response=[0.5,1,0.2,0.1,0.05,0.01]

N = 1000
p = 0.5  # Probabilidad de obtener un 1

x=generar_proceso_bernoulli(N,p)

#Calculo la respuesta del canal de comuniaciones
y=np.convolve(x,impulse_response)
#x=np.concatenate((x,np.zeros(d)))
d=len(impulse_response)
#print(y)


#Aplico el algoritmo LMS
estimaciones= np.zeros((d,len(x)-d))
error1= np.zeros(len(x)-d)
D=d-1
x_corrida=x[D:-1]
print("Largo de x_corrida",x_corrida.shape)

estimaciones,error1=LMS(y,d,x_corrida,np.zeros(d))


plt.figure(figsize=(10, 6))

plt.plot(abs(error1)**2, linestyle='-')

plt.title('Error del ecualizador en dB')
plt.xlabel('Iteracion del LMS')
plt.ylabel('Valor')
plt.grid(True)


print("Largo de las estimaciones",estimaciones.shape)


# Fase de la respuesta en frecuencia
# Crear un gráfico

plt.figure(figsize=(10, 6))

for i in range(len(estimaciones[:,0])):
    plt.plot(estimaciones[i,:], linestyle='-')

plt.title('Coeficientes estimados del ecualizador ')
plt.xlabel('Iteracion del LMS')
plt.ylabel('Valor')
plt.grid(True)
#
## Graficar líneas horizontales para cada valor en el vector
#for i in impulse_response:
#    plt.axhline(y=i, color='r', linestyle='-')
#
#plt.ylim(0, max(impulse_response)*1.1)
plt.show()


#Ploteo la respuesta del canal y la del canal estimado con el LMS
w, h = freqz(impulse_response)
w_estim,h_estim=freqz(estimaciones[:,-1])

ecualizador_estimado= estimaciones[:,-1]

print(ecualizador_estimado)


plt.figure(figsize=(10, 6))

plt.plot(w_estim,20*np.log(h_estim), linestyle='-')

plt.title('Respuesta en frecuencia dle ecualizador')
plt.xlabel('Frecuencia')
plt.ylabel('Amplitud')
plt.grid(True)
plt.show()

w_conjunto,h_conjunto=freqz(np.convolve(impulse_response,estimaciones[:,-1]))

plt.figure(figsize=(10, 6))

plt.plot(w_conjunto,20*np.log(h_conjunto), linestyle='-')

plt.title('Respuesta en frecuencia del ecualizador convolucionado con el canal')
plt.xlabel('Frecuencia')
plt.ylabel('Amplitud')
plt.grid(True)
plt.show()








##------------------------------------------------------------------------------------------------------------





largo_ecualizador=8#Largo del filtro LMS estimado
numero_realizaciones=500
corrimiento=8   #Retardo
largo=1000

x_original=np.zeros(largo)

y_c=np.zeros(largo+len(impulse_response)-1)

z=np.zeros(largo + len(impulse_response) - 1 + largo_ecualizador - 1)
error_realizacion=np.zeros(largo)
retardos=[1,2,3,4,5,6,7,8,9]

j_total=np.zeros((len(retardos),largo+len(impulse_response)))

for j in range(len(retardos)):
    j_total[j],ecualizador_optimo=iteracion_Jn(numero_realizaciones,largo,impulse_response,retardos[j],ecualizador_estimado,p,largo_ecualizador)

plt.figure()
for i in range(len(retardos)):
    plt.plot(20 * np.log(j_total[i]), label=f'Jn con retardo D={i+1} muestras')
plt.legend()
plt.title('J(n) en funcion de la realizaciones para distintos retardos')
plt.xlabel('Muestra')
plt.ylabel('Amplitud')
plt.grid()
plt.show()

entrada_canal=generar_proceso_bernoulli(largo,p)
entrada_desplazada=entrada_canal[8:-1]

salida_ecualizador=np.convolve(entrada_canal,ecualizador_optimo)
print("Largo de la entrada desplazasa",entrada_desplazada.shape)
print("Largo de la salida del ecualizador",salida_ecualizador.shape)

plt.figure()

plt.plot(salida_ecualizador,'r')
plt.plot(entrada_desplazada,'b')

plt.legend()
plt.title('J(n) en funcion de la realizaciones para distintos retardos')
plt.xlabel('Muestra')
plt.ylabel('Amplitud')
plt.grid()
plt.show()


w_conjunto, h_conjunto = freqz(np.convolve(impulse_response,ecualizador_optimo))
w_canal, h_canal = freqz(impulse_response)
w_ecualizador, h_ecualizador = freqz(ecualizador_optimo)


plt.figure(figsize=(10, 6))
plt.plot(w_conjunto,20*np.log(h_conjunto), linestyle='-' ,label='Conjunta')
plt.plot(w_canal,20*np.log(h_canal), linestyle='-',label='Canal')
plt.plot(w_ecualizador,20*np.log(h_ecualizador), linestyle='-',label='Ecualizador')
plt.title('Respuesta en frecuencia conjunta del ecualizador y el canal')
plt.legend()
plt.xlabel('Frecuencia')
plt.ylabel('Amplitud')
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 4))
plt.plot(np.convolve(impulse_response,ecualizador_optimo), marker='o')
plt.title('Respuesta impulsiva de h(n)*w(n) (Conjunta)')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.grid(True)
plt.show()

