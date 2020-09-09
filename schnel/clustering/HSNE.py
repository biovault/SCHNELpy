# pylint:disable-all

from scipy.sparse import csc_matrix, lil_matrix
import numpy as _np
import warnings
import igraph as ig
import leidenalg


class HSNE:
    def __init__(self, num_scales):
        """
        Initialization function for HSNE class.

        :param num_scales: number of scales
        """
        # Number of scales in hierarchy including datascale
        self.num_scales = num_scales
        # Scales which are at index 0 a datascale and the rest are subscales
        self.scales = [None] * num_scales
        self._index = -1

    def __str__(self):
        return "HSNE hierarchy with %i scales" % self.num_scales

    def __getitem__(self, index):
        return self.scales[index]

    def __setitem__(self, index, value):
        self.scales[index] = value

    def __iter__(self):
        return self

    def __next__(self):
        if self._index == self.num_scales - 1:
            self._index = -1
            raise StopIteration
        self._index += 1
        return self.scales[self._index]

    def get_topscale(self):
        """
        Getter for the top scale of the hierarchy.

        :return: first scale
        """
        return self.scales[0]

    def get_datascale_mappings(self, scalenumber):
        """
        Generates data scale mappings between the most representative landmarks and data vectors at a given scale.

        :param scalenumber: scale of desired mapping
        :return: dictionary of mappings, keys are enumerated and values are respective representative landmarks
        """
        if scalenumber <= 0:
            raise ValueError("Can't generate mapping for complete dataset, only scales get clustered")
        if scalenumber > self.num_scales:
            raise ValueError("Scale doesn't exist, object has %i scales" % self.num_scales)
        maps = None
        for scale in self.scales[1:scalenumber + 1]:  # Don't include datascale
            if maps is None:
                maps = dict.fromkeys(range(self.get_topscale().size))
                for idx, value in enumerate(scale.best_representatives):
                    maps[idx] = value
            else:
                for key in maps:
                    maps[key] = scale.best_representatives[maps[key]]
        return maps

    def get_map_by_cluster(self, scalenumber, clustering):
        """
        Maps a given cluster at a given scale.

        :param scalenumber: scale to create map at
        :param clustering: membership results of clustering
        :return: map between data and cluster assignment
        """
        if len(clustering) != self.scales[scalenumber].area_of_influence.shape[1]:
            raise ValueError("Number of labels does not match number of landmarks in scale")
        if scalenumber <= 0:
            raise ValueError("Can't generate mapping for complete dataset, only scales get clustered")
        for scale in self.scales[scalenumber:0:-1]:  # Don't include datascale
            new_aoi = lil_matrix((scale.area_of_influence.shape[0],
                                  len(set(clustering))))
            for i, label in enumerate(_np.unique(clustering)):
                new_aoi[:, i] = scale.area_of_influence * [[1] if label == x else [0] for x in clustering]
            clustering = csc_matrix(new_aoi).argmax(axis=1).A1
        return clustering

    def cluster_scale(self, scalenumber, prop_method='cluster'):
        """
        Clusters data using the Leiden algorithm on a given scale.

        :param scalenumber: scale that clustering should be applied to
        :param prop_method: label or cluster. Cluster returns cluster labels and label labels data scale mapping.
        :return: Clustering results. Either label or cluster.
        """
        if scalenumber == 0:
            warnings.warn("Warning: You are about to cluster the full dataset, this might take a very long time")
            return self.run_louvain(scalenumber)
        elif scalenumber >= self.num_scales:
            raise ValueError("Scale doesn't exist, object has %i scales" % self.num_scales)
        if prop_method == 'cluster':
            membership = self.run_louvain(scalenumber)
            return self.get_map_by_cluster(scalenumber, membership)
        elif prop_method == 'label':
            membership = self.run_louvain(scalenumber)
            mapping = self.get_datascale_mappings(scalenumber)
            return _np.asarray(membership)[list(mapping.values())]


        else:
            raise ValueError("Invalid method, options are 'label' or 'cluster'")

    def run_louvain(self, scalenumber):
        """
        Runs the Leiden algorithm on a given scale.

        :param scalenumber: scale to cluster on
        :return: membership of data
        """
        sources, targets = self.scales[scalenumber].tmatrix.nonzero()
        edgelist = list(zip(sources.tolist(), targets.tolist()))
        G = ig.Graph(edgelist)
        G.es['weight'] = self.scales[scalenumber].tmatrix.data
        return leidenalg.find_partition(G, leidenalg.ModularityVertexPartition, weights=G.es['weight']).membership


class DataScale:
    def __init__(self, num_scales, tmatrix=None):
        """
        Initialization function for DataScale class.
        HSNE data level object contains only the transition matrix.

        :param num_scales: number of scales
        :param tmatrix: transition matrix
        """
        self.tmatrix = tmatrix
        self.size = tmatrix.shape[0]
        self.datapoints = [idx for idx in range(self.size)]
        self.scalenum = 0
        self.num_scales = num_scales

    def __str__(self):
        return "HSNE datascale %i with %i datapoints" % (self.scalenum, self.size)


# HSNE L(s-x) scales contain many mappings


class SubScale:
    def __init__(self, scalenum, num_scales, tmatrix,
                 lm_to_original, lm_to_previous, lm_weights, previous_to_current, area_of_influence):
        """
        Initialization function for SubScale class.

        :param scalenum: scale number
        :param num_scales: number of scales
        :param tmatrix: transition matrix
        :param lm_to_original: landmarks to original data points
        :param lm_to_previous: landmarks to data points in previous scale
        :param lm_weights: landmark weights
        :param previous_to_current: previous landmarks to current landmarks
        :param area_of_influence: area of influence
        """
        # The transition matrix / graph
        self.tmatrix = tmatrix
        # Number of landmarks in scale
        self.size = tmatrix.shape[0]
        self.datapoints = [idx for idx in range(self.size)]
        # Scalenumber
        self.scalenum = scalenum
        # NUmber of scales in hierarchy
        self.num_scales = num_scales
        # Which landmark is which original datapoint
        self.lm_to_original = lm_to_original
        # Which landmark is which datapoint in the previous scale (reduntant
        # with lm_to_original on scale 1.
        self.lm_to_previous = lm_to_previous
        # LM Weights is equal to the sum of AOI columns
        self.lm_weights = lm_weights
        # Which landmark on previous scale is landmark on current scale
        self.previous_to_current = previous_to_current
        # Comes in as S x S where all columns > S-1 are 0's
        # Cast to csc to efficiently slice all columns outside range S-1
        self.area_of_influence = csc_matrix(area_of_influence)[:, :self.size]
        # The best representative landmark in scale S  for each point in scale S-1 is
        # the node that was visited most often e.g. has the highest value in its row
        # in area_of_influence.
        self.best_representatives = self.area_of_influence.argmax(axis=1).A1

    def __str__(self):
        return "HSNE subscale %i with %i datapoints" % (self.scalenum, self.size)
