# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:33:11 2019

@author: gordon.garisch
"""

from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
import numpy as np
import matplotlib.pyplot as plt 

def main():
    
    # Load boston data
    wine = load_wine()
    
    # Extract features
    X = wine.data

    # Set range of number of clusters
    k = np.arange(1,20,dtype=int)
    
    # Determine the total within cluster distance (inertia) for each k
    inertia = [KMeans(n_clusters=i,random_state=0).fit(X).inertia_ for i in k]
    
    # Create plot of intertia vs. number of clusters
    plt.cla()                        #Remove existing plots
    plt.plot(k, inertia)             #Plot intertia vs. k
    plt.xlabel('Number of clusters') #Add x label
    plt.ylabel('Inertia')            #Add y label
    plt.title('Reduced Fitting Error (Inertia) Per Cluster') #Add title
    plt.show()
    
if __name__ == "__main__":
    main()
