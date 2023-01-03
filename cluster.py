import numpy as np
from sklearn.cluster import AgglomerativeClustering as aglo
from scipy.spatial.distance import squareform
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import ete3
from collections import defaultdict
import random
import pandas as pd
import pickle




def similarity(rules, r1, r2):
    # |intersection r1,r2|*2 / (|r1|+|r2|)

	intersect = rules[r1].intersection(rules[r2])
	return 2 * len(intersect) / (len(rules[r1]) + len(rules[r2]))


def dissimilarity(rules, r1, r2):
	return 1 - similarity(rules, r1, r2)



def dist_matr_gen(rules):
	rules_num = len(rules)
	return np.array([[dissimilarity(rules, i, j) for i in range(rules_num)] for j in range(rules_num)])


def cluster(n, dist_matr):
	if dist_matr is None:
		dist_matr = dist_matr_gen()
	
	clustering = aglo(n_clusters=n, distance_threshold=None, linkage='average', compute_distances=True).fit(dist_matr)
	return clustering


def visualize(dist_matr):
	Z = linkage(dist_matr,method='average')
	dendrogram(Z)
	plt.show()


def plot_dendogram(model):
	print(model.children_, model.labels_, model.distances_)
	counts = np.zeros(model.children_.shape[0])
	n_samples = len(model.labels_)
	for i, merge in enumerate(model.children_):
		current_count = 0
		for child_idx in merge:
			if child_idx < n_samples:
				current_count += 1  # leaf node
			else:
				current_count += counts[child_idx - n_samples]
		counts[i] = current_count

	linkage_matrix = np.column_stack(
		[model.children_, model.distances_, counts]
	).astype(float)



	print("link:", linkage_matrix)
    # # Plot the corresponding dendrogram
	dendrogram(linkage_matrix)

	plt.show()