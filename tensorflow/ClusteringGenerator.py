from GeneratorInterface import GeneratorInterface

import numpy as np
from sklearn.cluster import KMeans


class ClusteringGenerator(GeneratorInterface):
    def __init__(self, clusters):
        self.kmeans = KMeans(n_clusters=clusters)
        self.num_clusters = clusters
    
    def initialize(self, encodings, *args, **kwargs): # TODO: Fix hack
        self.encodings = encodings

    def train(self, **kwargs):
        # Fits and then predicts each of the encodings
        self.labels = self.kmeans.fit_predict(self.encodings)
        clusters = [[] for i in range(self.num_clusters)]

        for i in range(len(self.labels)):
            clusters[self.labels[i]].append(self.encodings[i])

        self.stdev_clusters = []

        for i in range(self.num_clusters):
            self.stdev_clusters.append(np.std(np.array(clusters[i]), axis=0))

    # TODO: Implement save and load functions for k-means; for now, I'm not because
    # training should be really easy.
    def save(self):
        pass

    def load(self):
        pass

    def generate(self, number):
        # We pick a cluster center weighted by the number of points at each cluster
        # Then we sample from a normal distribution centered at the cluster and with STDEV
        # the STDEV of the cluster's points
        
        generated = []

        for i in range(number):
            index = np.random.choice(range(len(self.encodings)))
            label = self.labels[index]
            center = self.kmeans.cluster_centers_[label]

            noise = np.random.normal(size=self.encodings[0].shape)

            scaled_noise = np.multiply(noise, self.stdev_clusters[label])
            generated.append(scaled_noise + center)

        return np.array(generated)


