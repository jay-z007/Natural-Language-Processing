UNKNOWN = "UNK"
ROOT = "ROOT"
NULL = "NULL"
NONEXIST = -1


max_iter = 1001
batch_size = 10000

## The length of this list corresponds to the number of hidden layers andtheir sizes
## and the values correspond to the size of each layer
## For eg, [200 ,200] -> 2 hidden layers with 200 nodes each
hidden_size = [200] 

embedding_size = 50
learning_rate = 0.1
display_step = 100
validation_step = 200
n_Tokens = 48
lam = 1e-8

"""
Configuration for the best model 

max_iter = 501
batch_size = 10000
hidden_size = [200] 
embedding_size = 50
learning_rate = 0.003
display_step = 100
validation_step = 500
n_Tokens = 48
lam = 1e-8
"""