# Quick-Clustering-program
A simple clustering program that finds clusters in most kinds of data

QuickClustering.py


This is clustering program in Python 3.7 that take multidimensional data as a input, visualizes it and tries to find clusters within the data. 
Currently it is quite barebones and only supports k-Nearest-Neighbors clustering algoritm. In the future I hope to add additional algorithms
and have the program to try to select the best one based on the data.

HOW TO USE THIS PROGRAM

1. Preprocess data
  - The data has to only include numerical values except if it's the class of a particular instance of data
  - The class has to be the rightmost column
  - Each column has to have a label
  - Filetypes that are supported are .txt, .csv and .xlsx. Text files might have some difficulties if the delimiters are weird

2. Loading the data to the program
  - Have the data file in the same folder as the program and instert it's name and filetype to the reader function
  - If the instances of data do not have a class of their own then 'classes_exist' has to be set to False
  
3. Run the program

4. Output
  -The pairplot gives a scatterplot for all combinations of the data columns. This can give insight into the data.
  -Program does a z-score normalization to ensure that no data column weighs more than the others based on its absolute value
  -The program prints the three best silhouette scores that measure how well the data fits into that amount of clusters
  -The matching k_values are printed which gives the best amount of clusters for that data
  -Principal component analysis is performed for the Z-score normalized data and it's plotted. If data has different classes their color is 
    based on the class of that data point
  -The amount of clusters that had the best silhouette score is plotted as well


Dataset 'dim6.txt' is from University of Joensuu, Finland. They have a clustering basic benchmark dataset repository which was used for this
project. http://cs.joensuu.fi/sipu/datasets/
  
'iris_data.csv' was acquired from 'Data Analysis and Knowledge Discovery' course at University of Turku by Prof. Antti Airola. 
 
