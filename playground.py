import numpy as np

from knn.distance_metrics import SimiliarityMatrix
from knn.knn import kNN

example = np.array([[7, 6, 7, 4, 5, 4],
                        [6, 7, 0, 4, 3, 4],
                        [0, 3, 3, 1, 1, 0],
                        [1, 2, 2, 3, 3, 4],
                        [1, 0, 1, 2, 3, 3]])
user_sim = SimiliarityMatrix(example, axis=0)
user_sim.fit()
item_sim = SimiliarityMatrix(example, axis=1)
item_sim.fit()
classificator = kNN(example, user_sim.similarity, item_sim.similarity, user_sim.mean,k=2)