# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:39:20 2015

@author: enrique
"""


from scipy import ndimage
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io, color, data
import sys
import os
from PIL import Image
from pylab import ion, ioff
from interactive_only import*


def funBacktracking(matrixM):

    #print "matrixM.shape", matrixM.shape
    #M1=np.empty((matrixM.shape[0],matrixM.shape[1]),dtype=float)
    m2l=np.zeros((matrixM.shape[0],2),dtype=int)

    #print "m2l", m2l.shape
    
    #print "M1", matrixM.shape[0]
    
    minimoFila = np.min(matrixM[matrixM.shape[0]-1,:])
    pos = np.where(minimoFila==matrixM[matrixM.shape[0]-1,:])
    #print "minimo", minimoFila, "posicion", pos
    
    fila,columna=matrixM.shape[0]-1,pos[0][0]
    
    #print "dato", matrixM[fila,pos]    
    #print "fila,columna", fila, columna
    filaRecorrido=fila
    while(filaRecorrido!=-1):
        m2l[fila,0]=fila
        m2l[fila,1]=columna
        #m2l[fila,2]=minimoFila
        if(columna==0):
            #print "==========Start=========="
            minimoFila = min(matrixM[fila-1,columna],matrixM[fila-1,columna+1]) 
            pos = np.where(minimoFila==matrixM[fila-1,columna:columna+2]) 
            fila=fila-1
        elif(columna==(matrixM.shape[1]-1)):
            #print "==========Final=========="            
            minimoFila = min(matrixM[fila-1,columna-1],matrixM[fila-1,columna])  
            pos = np.where(minimoFila==matrixM[fila-1,columna-1:]) 
        else:
            #print "==========Medio=========="
            #print "Fila, columna",fila,columna
            #print "Fila",matrixM[fila-1,(columna-1):(columna+2)]
            minimoFila = min(matrixM[fila-1,columna-1],matrixM[fila-1,columna],matrixM[fila-1,columna+1])
            pos = np.where(minimoFila==matrixM[fila-1,(columna-1):(columna+2)]) 
            fila,columna=fila-1,columna+(pos[0][0]-1)
        #print "----MinimoFila",minimoFila
        #print "==========Pos==============",pos
        #fila,columna=fila-1,columna+(pos[0][0]-1)
        #print "Fila, Columna",fila, columna
        filaRecorrido-=1
        
    #○print "m2l", m2l
    
    return m2l
    

#La funcion nos marca el rayo en la imagen
def markPath(mat, path, mark_as='red'):
    assert mark_as in ['red','green','blue','black','white']
    
    if len(mat.shape) == 2:
        mat = color.gray2rgb(mat)
    
    ret = np.zeros(mat.shape)
    ret[:,:,:] = mat[:,:,:]
    
    # Preprocess image
    if np.max(ret) < 1.1 or np.max(ret) > 256: # matrix is in float numbers
        ret -= np.min(ret)
        ret /= np.max(ret)
        ret *= 256
    
    # Determinate components
    if mark_as == 'red':
        r,g,b = 255,0,0
    elif mark_as == 'green':
        r,g,b = 0,255,0
    elif mark_as == 'blue':
        r,g,b = 0,0,255
    elif mark_as == 'white':
        r,g,b = 255,255,255
    elif mark_as == 'black':
        r,b,b = 0,0,0

    # Place R,G,B
    for i in path:
        ret[i[0],i[1],0] = r
        ret[i[0],i[1],1] = g
        ret[i[0],i[1],2] = b
    return ret.astype('uint8')    

#Nos genera indices de vector 1D a partir de indices 2D
def generateIndexes(index2d,nColumns):
    indexes=[]    
    for i in index2d:
        index=i[0]*nColumns+i[1]
        indexes.append(index)
    return indexes
    
def getValues(channel,path):
    values=[]
    for point in path:
        values.append(channel[point[0],point[1]])
    return values
    
#Funcion que nos reduce una imagen eliminando las lineas con bajo gradiente
def imgReduce(img,path,reduceLines):
    nChannel=img.shape[2]
    reducedImg=np.zeros((img.shape[0]-reduceLines[0],img.shape[1]-reduceLines[1],img.shape[2]),dtype=np.uint8)
    nColumns=img.shape[1]
    #reducedImg=np.array(img[:,:,i])
    indexes=generateIndexes(path,nColumns)
    for i in range(nChannel):
        reducedImg[:,:,i]=np.reshape(np.delete(img[:,:,i],indexes),(reducedImg.shape[0],reducedImg.shape[1]))
    return reducedImg 
    
#Funcion que nos sintetiza una imagen añadiendo una linea de informacion
def imgExtend(img,path,extendLines):
    nChannel=img.shape[2]
    nColumns=img.shape[1]
    extendedImg=np.zeros((img.shape[0]+extendLines[0],img.shape[1]+extendLines[1],img.shape[2]),dtype=np.uint8)
    #reducedImg=np.array(img[:,:,i])
    indexes=generateIndexes(path,nColumns)
    for i in range(nChannel):
        values=getValues(img[:,:,i],path)
        extendedImg[:,:,i]=np.reshape(np.insert(img[:,:,i],indexes,values),(extendedImg.shape[0],extendedImg.shape[1]))
    return extendedImg

#Funcion para calcular el gradiente de una imagen para la sintesis o reduccion
def calcGradient(img): 
    imgScaleGray = color.rgb2gray(img)
    matrix_double=np.array(imgScaleGray).astype("double")
    
    gX,gY=np.gradient(matrix_double) #imgScaleGray
    return np.abs(gX)+np.abs(gY)

#Funcion que nos devuelve la linea que se puede eliminar de una imagen o la linea para para hacer la 
#sintesis
def generateDelLines(img):
    lines=[]
    gXY=calcGradient(img)
    
    size_Y=np.shape(gXY)[0]
    size_X=np.shape(gXY)[1]    
    M=np.zeros([size_Y,size_X],dtype=float) #type(gXY[0,0])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if(i==0):
                M[i,j] = gXY[i,j]
            else:
                if(j >= M.shape[1]-1):
                    M[i,j] = gXY[i,j]+min(M[i-1,j-1],M[i-1,j])
                else:
                    M[i,j]=gXY[i,j]+min(M[i-1,j-1],M[i-1,j],M[i-1,j+1])
                            
        
    lines=funBacktracking(M)
    return lines
    
    
#Seam carving para la reduccion de una imagen
def seamCarvingReduction(path):
    print "=========================================="
    print "           Seam Carving Reduction         "   
    print "=========================================="    
    #img = mpimg.imread("torre3.jpg")
    #img = mpimg.imread(path+"\iberia.jpg")
    #img=mpimg.imread("agbar.jpg")    
    img=mpimg.imread("countryside.jpg")    
    reduceSize=[0,50]#Lineas a eliminar
    
    plt.show()
    plt.imshow(img)
    for nlines in reduceSize:
        for i in range(nlines):
            delLines=generateDelLines(img)
            figuraMarcada=markPath(img, delLines, mark_as='red')
            img=imgReduce(img,delLines,[0,1])
            plt.show()
            plt.imshow(figuraMarcada)
            plt.title("Rayo "+str(i))
    plt.show()
    plt.imshow(img)    
    plt.title("Imagen reducida")
     
    
#Seam Carving para la sintesis de una imagen
def seamCarvingSintesis(path):
    print "=========================================="
    print "           Seam Carving Syntesis         "   
    print "=========================================="    
    img = mpimg.imread("countryside.jpg")
    #img = mpimg.imread("iberia.jpg")    
    #img = mpimg.imread("torre3.jpg")    
    reduceSize=[0,50]
    
    plt.show()
    plt.imshow(img)
    for nlines in reduceSize:
        for i in range(nlines):
            duplicateLines=generateDelLines(img)
            figuraMarcada=markPath(img, duplicateLines, mark_as='red')
            img=imgExtend(img,duplicateLines,[0,1])
            plt.show()
            plt.imshow(figuraMarcada)
            plt.title("Rayo "+str(i))
    plt.show()
    plt.imshow(img)  
    plt.title("Imagen sintetizada")    
    
def main():
    raiz=os.getcwd()
    plt.close("all")
    seamCarvingReduction(raiz)
    seamCarvingSintesis(raiz)
main()
    