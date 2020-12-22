import timeit

import numpy as np
import pandas as pd
import scipy.stats as st
import yep
from BanditPAM import KMedoids

times = []

def runtime_and_profile(dataset_size, k=5):
    print("running mnist-{}k-parallelized-cache-k-{}".format(dataset_size, k))
    for i in range(2):
        if i == 9:
            yep.start("mnist-{}k-parallelized-cache-k-{}".format(dataset_size, k))
        # Load the 1000-point subset of MNIST and calculate its t-SNE embeddings for visualization:
        X = pd.read_csv('data/MNIST-{}k.csv'.format(dataset_size), sep=' ', header=None).to_numpy()

        # Fit the data with BanditPAM:
        start = timeit.default_timer()
        kmed = KMedoids(n_medoids = k, algorithm = "BanditPAM")
        kmed.fit(X, 'L2', k, "mnist_log")
        elapsed = start - timeit.default_timer()
        times.append(-elapsed)
        print(elapsed)
        print(kmed.final_medoids)

        if i == 9:
            yep.stop()

    print("mnist-{}k-parallelized-cache-k-{}".format(dataset_size, k), st.t.interval(0.95, len(times)-1, loc=np.mean(times), scale=st.sem(times)))

for val in [70]:
    runtime_and_profile(val)

# sizes = [1, 10, 20, 70]
# pool = mp.Pool(processes=len(sizes))
# pool.map(runtime_and_profile, sizes)
