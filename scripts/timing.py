# Use this script as a hacky way to verify BanditPAM is multithreaded
# Should complete in <= 3 seconds

import time
import numpy as np
from BanditPAM import KMedoids
X = np.loadtxt('data/MNIST-1k.csv')
kmed = KMedoids(n_medoids = 5, algorithm = "BanditPAM")
start = time.time()
kmed.fit(X, 'L2', 'gmm_log')
print(time.time() - start, "seconds")
print(kmed.medoids)
