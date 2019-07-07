import numpy as np

A = np.array([[1, 2, 3, 0],
              [0, 0, 2, 0],
              [2, 1, 0, 0],
              [0, 2, 2, 0]])

values = np.unique(A)[1:] #0 nicht vergeben


#fuer jeden Wert aus values:
#hier beispielsweise 1
A1 = np.full_like(A,1)
A1[(A-1).nonzero()]=0

A2 = np.full_like(A,2)
A2[(A-2).nonzero()]=0

A3 = np.full_like(A,3)
A3[(A-3).nonzero()]=0

#Anzahl der belegten Reihen mit Wert i
row_count = np.sum(A2,axis=1)/2