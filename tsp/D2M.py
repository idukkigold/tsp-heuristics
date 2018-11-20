import numpy as np
from sklearn.decomposition import PCA


C=[[0,3,4,6,8,9,8,10],
   [3,0,5,4,8,6,12,8],
   [4,5,0,2,2,3,5,7],
   [6,4,2,0,3,2,5,4],
   [8,8,2,3,0,2,2,4],
   [9,6,3,2,2,0,3,2],
   [8,12,5,5,2,3,0,2],
   [10,9,7,4,4,2,2,0]]

pca = PCA(n_components=2)
X3d = pca.fit_transform(C)
print(X3d)

