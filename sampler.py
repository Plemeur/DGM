import numpy as np


class Sampler:
    def __init__(self, bound_list=[], oversampling=[]):
        """ Definition of a sampler over a domain, the boundary list length have to be pair
            Oversampling allow for stepping out of the boundaries """
        self.bound_list = bound_list
        self.oversampling = oversampling

    def get_sample(self, N):
        dim = len(self.bound_list)//2
        samples = np.empty([N, dim], dtype=np.float32)

        # for ease of writing
        bl = self.bound_list
        os = self.oversampling

        for j in range(dim):
            i = j*2
            samples[:, j] = np.random.uniform(low=bl[i] - os[i]*(bl[i+1]-bl[i]),
                                              high=bl[i+1] + os[i]*(bl[i+1]-bl[i]),
                                              size=N)

        return samples



#s1 = Sampler([0, 1, 0, 1], [0.5, 0, 0.5, 0.5])
#sample1 = s1.get_sample(5)
#print(sample1)