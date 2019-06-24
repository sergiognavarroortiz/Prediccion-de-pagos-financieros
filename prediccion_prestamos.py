# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:06:58 2019

@author: Esteban Arrieta
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# TensorFlow backend.
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical

# Librerias de Deep Learning 
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight

from sklearn.model_selection import train_test_split

import warnings
from pandas import ExcelWriter

class Process_Analytical():
    
    def __init__(self, data):
        self.data = data
        
    def resumen(self, columns):
        for col in columns:
            print(self.data[col].describe())
    
    def limit_credit(self):
        # Límite de crédito (Nombre de la característica: 'LIMIT_BAL')
        """
        Regla de Sturges" -> Regla que consiste en determinar el numero de 
        clases a particionar un histograma c = 1 + log2(M), M = Longitud del dataset
        """
        total = self.data.shape
        k = int(np.ceil(1 + np.log2(total[0])))
        bins = k #30
        
        # Plots
        plt.figure(figsize = (8,5))
        plt.hist(self.data.LIMIT_BAL, bins = bins, color = 'm', label = 'Total', alpha = 0.5)
        plt.hist(self.data.LIMIT_BAL[self.data['default.payment.next.month'] == 1], bins = bins, color = 'b', label = 'Default')
        
        # Labels
        plt.xlabel('Limite de credito (NT dolar)')
        plt.ylabel('Numero de cuentas')
        plt.title('Limite de credito ',fontweight="bold", size=14)
        plt.legend(fontsize = 'x-large')
        plt.show()

    def genero(self):
        # Género (Nombre de la característica: 'SEX')
        total_Gen = dict(self.data.groupby('SEX').size())
        
        #men = total_Gen[1]
        men_default = self.data['SEX'][(self.data['SEX'] == 1) & (self.data['default.payment.next.month'] == 1)].count()
        
        #women = total_Gen[2]
        women_default = self.data['SEX'][(self.data['SEX'] == 2) & (self.data['default.payment.next.month']==1)].count()
        
        default_sex = [men_default, women_default]
        gender = [1,2]
        
        # Plots
        plt.figure(figsize = (8,5))
        plt.bar(gender, list(total_Gen.values()), color = 'm', alpha = 0.5, label = 'Total')
        plt.bar(gender, default_sex, color = 'b', alpha = 0.5, label = 'Default')        
        plt.xticks([1,2], ['Hombre','Mujer'])
        
        # Labels
        plt.ylabel('Numero de cuentas')
        plt.title('Genero ',fontweight = "bold", size = 12)
        plt.legend(fontsize = 'x-large')
        plt.show()
        
    def education(self):
        # Educación (Nombre de la característica: 'EDUCATION')
        total_edu = dict(self.data.groupby('EDUCATION').size())
        
        other = sum({v for k, v in total_edu.items() if k >= 4})
        total_edu = {k: v for k, v in total_edu.items() if k > 0 and k < 4}
        total_edu[4] =other
        
        #grad =  self.data['EDUCATION'][self.data['EDUCATION']==1].count()
        grad_default = self.data['EDUCATION'][(self.data['EDUCATION']==1)&(self.data['default.payment.next.month']==1)].count()
        
        #uni =  self.data['EDUCATION'][self.data['EDUCATION']==2].count()
        uni_default = self.data['EDUCATION'][(self.data['EDUCATION']==2)&(self.data['default.payment.next.month']==1)].count()
        
        #high =  self.data['EDUCATION'][self.data['EDUCATION']==3].count()
        high_default = self.data['EDUCATION'][(self.data['EDUCATION']==3)&(self.data['default.payment.next.month']==1)].count()
        
        #other =  self.data['EDUCATION'][self.data['EDUCATION'] > 3].count()
        other_default = self.data['EDUCATION'][(self.data['EDUCATION'] > 3)&(self.data['default.payment.next.month']==1)].count()
        
        #total_education = [grad, uni, high, other]
        default_education = [grad_default,uni_default,high_default, other_default]
        
        # Plots
        degree = [1,2,3,4]
        plt.figure(figsize = (8,5))
        plt.bar(degree, list(total_edu.values()), color = 'm', alpha = 0.5, label = 'Total')
        plt.bar(degree, default_education, color = 'b',alpha = 0.5, label = 'Default')
        
        # Labels
        plt.xticks([1,2,3,4],['Graduados','Universidad','Preparatoria','Otra'])
        plt.ylabel('Numero de cuentas');plt.title('Fig.3 : Educación ',fontweight="bold", size=12)
        plt.legend(fontsize = 'x-large')
        plt.show()

    def marital_status(self):
        # Estado Matrimonial (Nombre de la característica: 'MARRIAGE')
        total_marital = dict(self.data.groupby('MARRIAGE').size())
        total_marital = {k: v for k, v in total_marital.items() if k > 0 and k < 4}
        
        #married =  data['MARRIAGE'][data['MARRIAGE']==1].count()
        married_default = self.data['MARRIAGE'][(self.data['MARRIAGE'] == 1) & (self.data['default.payment.next.month'] == 1)].count()
        
        #single =  data['MARRIAGE'][data['MARRIAGE']==2].count()
        single_default = self.data['MARRIAGE'][(self.data['MARRIAGE'] == 2) & (self.data['default.payment.next.month'] == 1)].count()
        
        #other =  data['MARRIAGE'][data['MARRIAGE']==3].count()
        other_default = self.data['MARRIAGE'][(self.data['MARRIAGE'] == 3) & (self.data['default.payment.next.month'] == 1)].count()
        
        #total_marriage = [married, single, other]
        default_marriage = [married_default, single_default, other_default]
        
        # Plots
        status = [1,2,3]
        plt.figure(figsize = (8,5))
        plt.bar(status, list(total_marital.values()), color = 'm', alpha = 0.5, label = 'Total')
        plt.bar(status, default_marriage, color = 'b', alpha = 0.5, label = 'Default')
        
        # Labels
        plt.xticks([1,2,3], ['Casados','Solteros','Otros'])
        plt.ylabel('Numero de cuentas')
        plt.title('Estado civil ', fontweight = "bold", size = 12)
        plt.legend(fontsize = 'x-large')
        plt.show()
        
    def age(self):
        # Edad (Nombre de la característica: 'AGE')
        total = self.data.shape
        k = int(np.ceil(1 + np.log2(total[0])))
        bins = k #25
        
        # Plots
        plt.figure(figsize = (8,5))
        plt.hist(self.data['AGE'], bins = bins, color = 'm', label = 'Total',alpha = 0.5)
        plt.hist(self.data.AGE[self.data['default.payment.next.month'] == 1], bins = bins, color='b',label = 'Default',alpha = 0.5)
        
        # Labels
        plt.xlabel('Edad')
        plt.ylabel('Numero de cuentas')
        plt.title('Edad ', fontweight = "bold", size = 12)
        plt.legend(fontsize = 'x-large')
        plt.show()
        
    def payment_status(self):
        # Estado de pago (Nombre de la característica: 'PAY_')
        features = self.data.columns.values
        plt.figure(figsize = (14,10))
        
        gs = gridspec.GridSpec(3,2)
        cont = 0
        
        plt.suptitle('Estado de pago',fontweight = "bold", fontsize = 22)
        for cn in features[6:12]:
            ax = plt.subplot(gs[cont])
            delay = np.zeros(12)
            delay_default = np.zeros(12)  
            for j in np.arange(0, 12):
                    delay[j] = self.data[cn][self.data[cn] == j-2].count()
                    delay_default[j] = self.data[cn][(self.data[cn] == j-2) & (self.data['default.payment.next.month'] == 1)].count()  
        
            month = [-2,-1,0,1,2,3,4,5,6,7,8,9]
            plt.bar(month, delay, color = 'c', alpha = 0.5, label = 'Total')
            plt.bar(month, delay_default, color = 'k', alpha = 0.5, label = 'Default')
        
            plt.xticks([-2,-1,0,1,2,3,4,5,6,7,8,9], ['0 Cuentas','A tiempo','Parcial','1','2','3','4','5','6','7','8','9'], size = 10, rotation = 50)
            ax.set_xlabel('Delay (month)')
            months = ['Septiembre','Agosto','Julio','Junio','Mayo','Abril']
            ax.set_title('Estado de pago ' + months[cont], fontweight = "bold", size = 12)
            ax.legend(fontsize = 'large')
            cont += 1
            
        plt.tight_layout()
        plt.subplots_adjust(top = 0.90)
        plt.show()
        
    def bill_amount(self):
        # Monto del extracto de factura (Nombre de la característica: 'BILL_AMT_')        
        total = self.data.shape
        k = int(np.ceil(1 + np.log2(total[0])))
        bins = k # 25
        
        features = self.data.columns.values
        plt.figure(figsize=(12,12))
        
        gs = gridspec.GridSpec(3,2)
        cont = 0
        
        plt.suptitle('Monto del estado de cuenta', fontweight = "bold", fontsize = 22)
        for cn in features[12:18]:
            ax = plt.subplot(gs[cont])
            #bins = k
            plt.hist(self.data[cn], bins = bins, color = 'purple',label = 'Total', alpha = 0.5)
            plt.hist(self.data[cn][self.data['default.payment.next.month'] == 1], bins = bins, color = 'lightgray', label = 'Default', alpha = 0.9)
        
            plt.xlabel('Monto (NT dolar)')
            plt.ylabel('Numero de cuentas')
            plt.xticks(rotation = 45)
            ax.set_yscale('log', nonposy = 'clip')
        
            months = ['Septiembre','Agosto','Julio','Junio','Mayo','Abril']
            ax.set_title('Monto del estado de cuenta en  ' + months[cont], fontweight = "bold", size = 12)
            #ax.set_xticklabels(self.data[cn], rotation = 60)
            ax.legend(fontsize = 'large')
            cont += 1          
        
        plt.tight_layout()            
        plt.subplots_adjust(top = 0.90)
        plt.show()

    def payment(self):
        # Cantidad del pago anterior (Nombre de la característica: 'PAY_AMT_')
        total = self.data.shape
        k = int(np.ceil(1 + np.log2(total[0])))
        bins = k # 25
        
        features = self.data.columns.values
        plt.figure(figsize = (12,12))
        
        gs = gridspec.GridSpec(3,2)
        cont = 0
        
        plt.suptitle('Importe del pago anterior',fontweight = "bold", fontsize = 22)
        for cn in features[18:24]:
            ax = plt.subplot(gs[cont])
            #bins = 25
            plt.hist(self.data[cn], bins = bins, color = 'lightblue', label = 'Total', alpha = 1)
            plt.hist(self.data[cn][self.data['default.payment.next.month'] == 1], bins = bins, color = 'k', label = 'Default', alpha = 0.5)
        
            plt.xlabel('Monto (NT dolar)')
            plt.ylabel('Numero de cuentas')
            plt.xticks(rotation = 45)
            ax.set_yscale('log', nonposy='clip')
            plt.ticklabel_format(style = 'sci', axis = 'x', scilimits = (0,0))
            months = ['Septiembre','Agosto','Julio','Junio','Mayo','Abril']
            ax.set_title('Importe del pago anterior en ' + months[cont], fontweight = "bold", size = 12)
            ax.legend()
            cont += 1      
        
        plt.tight_layout()
        plt.subplots_adjust(top = 0.90)
        plt.show()
        
    def plot_correlation(self):
        # Making correlation coefficients pair plot of all feature in order to identify degenrate features
        plt.figure(figsize = (15,15))
        ax = plt.axes()
        corr = self.data.drop(['ID'], axis = 1).corr()
        sns.heatmap(corr, vmax = 1, vmin = -1, square = True, annot = True, cmap = 'Spectral', linecolor = "white", linewidths = 0.01, ax = ax)
        ax.set_title('Grafico de coeficiente de correlación', fontweight = "bold", size = 30)
        plt.show()
        
class Prediction():
    
    def __init__(self, data):
        self.data = data
    
    def predictor(self):

        predictors = self.data.drop(['ID', 'default.payment.next.month'], axis = 1).as_matrix()
        predictors = StandardScaler().fit_transform(predictors)
        
        # Generar una matrix binaria en base a N clases de entrada (default.payment.next.month = [0 - No pago, 1 - Pago])
        # 2 clases => {0, 1} = array([1, 1, 0, 0, 0, ..., 1]) --> array([0. 1.], [0. 1.], [1. 0.], [1. 0.], [1. 0.],..., [0. 1.]])
        target = to_categorical(self.data['default.payment.next.month'])
        
        return predictors, target

class neuronal_network():
    
    def __init__(self, data, predictors, target, test):
        #super().__init__(data)
        self.data = data
        self.predictors = predictors
        self.target = target
        self.test = test
    
    def neuron(self):
        # Cálculo de la relación por defecto
        
        non_default = len(self.data[self.data['default.payment.next.month'] == 0])
        default = len(self.data[self.data['default.payment.next.month'] == 1])
        ratio = float(default/(non_default + default))
        #print('Default Ratio : ',ratio)
        
        #predictors, target = super().predictor()
        #print(predictors, target)
        
        # Ajuste del modelo
        n_cols = self.predictors.shape[1]
        early_stopping_monitor = EarlyStopping(patience = 2)
        class_weight = {0:ratio, 1:1-ratio}
        
        model = Sequential()
        # Capaz ocultas
        # Una capa de entreda (input_shape de numero de columnas) y dos capas ocultas de 25 nodos
        model.add(Dense(25, activation = 'relu', input_shape = (n_cols,)))
        model.add(Dense(25, activation = 'relu'))
        #model.add(Dense(20, activation='relu'))
        # Capa de salida (Cantidad de nodos de salida - Total de clases de salida -> (2 [0, 1]))
        model.add(Dense(2, activation = 'softmax'))
        
        # Compilacion y entrenamiento del modelo
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        model.fit(self.predictors, self.target, epochs = 20, validation_split = 0.3, callbacks = [early_stopping_monitor], class_weight = class_weight)
        
        # batch_size. Permite controlar el tamaño de cada grupo de datos que pasara atravez de la red
        #Las GPU más grandes podrían acomodar lotes más grandes. Recomiendo comenzar con 32 o 64 y subir desde allí.
        loss = model.evaluate(self.predictors, self.target, batch_size = 128)
        
        pred_model = model.predict(self.predictors, batch_size = 128)
        #print(model.summary())
        
        predicted_class = np.argmax(pred_model ,axis = 1)
        
        # Optimizar modelo
        #optimization = pd.DataFrame()
        #optimization['Hidden Layer'] = [1, 1, 1, 2, 2, 3]
        #optimization['Node per Layer'] = [25, 50, 100, 25, 50, 25]
        #optimization['val_loss'] = [0.1874, 0.1871, 0.1876, 0.1861, 0.1875, 0.1881]
        
        #Imprimir optimizacion del modelo
        #df = optimization.head(6)
        
        return loss, pred_model, predicted_class

#def finanzas():
    
warnings.filterwarnings("ignore")

# Cargando Datos
path = "C:/Users/Keitlan Arrieta/Documents/TRABAJO/LATBC/Perfiles financieros/Predicciones financieras completo/default-of-credit-card-clients-dataset/"
data = pd.read_csv(path + 'UCI_Credit_Card.csv', sep = ",")

train, test = train_test_split(data, test_size = 0.2)

analisis = Process_Analytical(data)

""" Graficas """
print(" >>> Proceso Analitico <<<")
analisis.limit_credit()
analisis.genero()
analisis.education()
analisis.marital_status()
analisis.age()
analisis.payment_status()
analisis.bill_amount()
analisis.payment()
analisis.plot_correlation()

pred = Prediction(data)
predictors, target = pred.predictor()

rn = neuronal_network(data, predictors, target, test)
loss, modelo, clase = rn.neuron()

print("\n >> Validacion: loss => {} Acc => {}\n >> Modelo: \n{}\n >> Clases: {}".format(loss[0], loss[1], modelo[0:5], clase[0:5]))

data["Prediccion"] = clase
val = []
for k, v in enumerate(data["default.payment.next.month"]):
    if v == clase[k]:
        val.append("Correcta")
    else:
        val.append("Incorrecta")
#        
data["Validar_Prediccion"] = val

writer = ExcelWriter(path + "prestamos.xlsx")
data.to_excel(writer, 'datos', index = False)
writer.save()

#del (loss, predictors, target, train, test)