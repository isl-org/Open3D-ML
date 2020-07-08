import numpy as np
import lib.python.nearest_neighbors as nearest_neighbors
import time

batch_size = 16
num_points = 81920
K = 16
pc = np.random.rand(batch_size, num_points, 3).astype(np.float32)

# nearest neighbours
start = time.time()
neigh_idx = nearest_neighbors.knn_batch(pc, pc, K, omp=True)
print(time.time() - start)


