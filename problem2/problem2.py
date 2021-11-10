# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 23:45:07 2021

@author: João
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model



# 2.1.2
#Cria a matriz de design atraves dos pontos de treino
def polynomialToMatrix(degree, x):

    polyMatrix=np.zeros((x.size, degree+1))

    for i in range(x.size):

        polyMatrix[i][0]= 1
        for j in range(1,degree+1):
            
            polyMatrix[i][j]= x[i]**(j)

    return polyMatrix



#Calcula os coeficientes do polinomio com base na matriz de design e no vector de resultados
def findBeta(x, y):

    xT=np.transpose(x)

    beta=np.matmul(np.linalg.inv(np.matmul(xT, x)), np.matmul(xT, y))

    return beta



#Calcula o resultado previsto pela funcao polinomial
def predict(beta, value, degree):

    prediction=beta[0][0]

    for i in range(degree):

        prediction+=beta[i+1]*((value)**(i+1))

    return  prediction



#Calcula o SSE com base na matriz de design, nos coeficientes do polinomio e no vector de resultados
def mse(x, y, beta):

    mseError=np.mean((y-np.matmul(x,beta))**2)

    return mseError



def main(data_x, data_y, degree, number):
    
    #Calcula a matrix de design
    polyMatrix=polynomialToMatrix(degree, data_x)

    #Calcula os coeficientes do polinomio que melhor ajusta aos dados
    if np.linalg.det(np.matmul(np.transpose(polyMatrix), polyMatrix))!=0:
        beta=findBeta(polyMatrix, data_y)
    else:
        print("Det=0, matriz nao inversora.")
        exit()

    #Calcula o SSE
    mseError=mse(polyMatrix, data_y, beta)

    #Imprime os coeficientes do polinomio e o SSE
    print("Coeficients= \n", beta)
    print("MSE=\n", mseError)

    #Desenha os pontos de treino na janela
    plt.plot(data_x, data_y, 'bx')

    #Desenha a recta que melhor ajusta os pontos de ordem P
    time=np.arange(-1,1, 0.01)
    amplitude=list()
    for i in range (int(2/0.01)):
        amplitude.append((predict(beta, time[i], degree))[0])
    plt.plot(time, amplitude , 'r-')


    title='Ajuste dos dados '+str(number)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    return beta





#2.1.5
#Carrega os dados

data_x = np.load("Xtrain_Regression_Part2.npy")
aux = np.load("Ytrain_Regression_Part2.npy")
xtest = np.load("Xtest_Regression_Part2.npy")


data_y=np.reshape(aux, (aux.size, 1))
degree=2

#beta=main(data_x, data_y, degree, "2a")

#Elimina outliers dos dados


#Calcula a matriz de design
polyMatrix=polynomialToMatrix(degree, data_x)

#Calcula o SSE
mseError=mse(polyMatrix, data_y, beta)
print("MSE2=\n", mseError)