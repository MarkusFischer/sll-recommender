import numpy as np

from utility.matrices import make_rows_mean_free


class kNN:
    def __init__(self, train, k, row_metric, column_metric, row_weight=1, col_weight=1,regression=False,
                 col_mean_usage=False, row_mean_usage=True):
        self.train_matrix = train
        mean_free, mean = make_rows_mean_free(train)
        self.train_matrix_mean_free = mean_free
        self.mean = mean
        self.k = k
        if (k <= 0):
            raise Exception(f"K value {self.k} is to small")#todo check if not int
        self.row_metric=row_metric
        self.column_metric=column_metric
        self.row_weight=row_weight
        self.col_weight=col_weight
        self.regression=regression
        self.col_mean_usage=col_mean_usage
        self.row_mean_usage=row_mean_usage
        pass

    def classify(self,u,i,dir=None): #todo vectorize
        if dir is None:
            pass
        elif dir == 0:
            neighbourhood = {}
            for j in range(0,self.train_matrix.shape[0]): #todo what happens when row is not given?
                if j == u: #todo better way
                    continue
                if self.train_matrix[j, i] == 0:
                    continue
                if self.row_mean_usage:
                    neighbourhood[j] = self.row_metric(u,j,self.train_matrix,self.mean)
                else:
                    neighbourhood[j] = self.row_metric(u, j, self.train_matrix)
            sorted_neighbourhood = sorted(neighbourhood.items(), key=lambda item: item[1],reverse=True)
            knearest = sorted_neighbourhood[:(self.k)]
            if self.row_mean_usage:
                s = np.empty((0,0))
                sim = 0
                for v in knearest:
                    np.append(s, self.row_metric(u,v[0],self.train_matrix,self.mean)*self.train_matrix_mean_free[v[0],i])
                    sim += np.abs(self.row_metric(u,v[0],self.train_matrix,self.mean))
                return self.mean[u] + (np.sum(s)/sim)
            else:
                pass
        elif dir == 1:
            pass
        else:
            raise Exception(f"Argument value {dir} not supported for parameter dir")