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
from math import log
import math

from cluster import *

import warnings


# -------------------------------------------------------------------------
# ---- load data ----

def read_rules(file_name='rules.pkl'):
	"""
	 Return exactly the pysubgroup output (a dataframe contain quality, subgroup, ...)
	"""
	return pd.read_pickle("pickle_data/" + file_name)


def read_id_set(file_name='id_set.pkl'):
	"""
		Return a dictionary {rule_ids:set_of_instances}
	"""

	with open('pickle_data/' + file_name, 'rb') as handle:
		return pickle.load(handle)


# -------------------------------------------------------------------------
# ---- sillh ----


def a_r(r, labels, clusters, dist_matr):
	cluster = labels[r]
	nomin = 0
	for i in clusters[cluster]:
		if i == r:
			continue
		nomin += dist_matr[i][r]

	denom = len(clusters[cluster]) - 1

	if denom == 0:
		return 0

	return nomin / denom


def b_r(r, labels, n_cluster, clusters, dist_matr):
	# cluster = labels[get_neighbor(r,n_cluster)]
	cluster = get_neighbor(labels[r], n_cluster)

	nomin = 0
	for i in clusters[cluster]:
		nomin += dist_matr[i][r]

	denom = len(clusters[cluster])

	# print(cluster,len(clusters[cluster]), nomin/denom, "XXXXXX")
	return nomin / denom


def sillh(labels, n_cluster, clusters, dist_matr, rules_num):
	res = 0
	for r in range(rules_num):
		b = b_r(r, labels, n_cluster, clusters, dist_matr)
		a = a_r(r, labels, clusters, dist_matr)
		if max(b, a) == 0:
			b = 1
		res_cur = (b - a) / max(b, a)
		res += res_cur

		if math.isnan(res_cur):
			print("shit", r, res_cur, max(a, b))
	# res += log(b-a) - log(max(b,a))
	# res += (b-a)

	# print(res)
	res /= rules_num
	return res


def get_neighbor(i, n_clusters):
	# should change to reflect the distances.
	if i % 2 == 1 or i == n_clusters - 1:
		return i - 1
	return i + 1


def create_clusters(labels):
	clusters = defaultdict(list)
	for j, clus in enumerate(labels):
		clusters[clus].append(j)
	return clusters


# -------------------------------------------------------------------------
# ---- main function ----

def find_best(silent=True):
	if silent:
		warnings.filterwarnings("ignore")

	rules = read_id_set()
	rules_num = len(rules)

	dist_matr = dist_matr_gen(rules)

	clusts = list()
	sills = list()
	y = [i for i in range(2, 20)]
	for i in y:
		clustering = cluster(i, dist_matr)
		clusts.append(clustering)
		n_cluster = i

		labels = clustering.labels_
		clusters = create_clusters(labels)

		sill = sillh(labels, n_cluster, clusters, dist_matr, rules_num)
		sills.append(round(sill,3))
		if not silent:
			print(sill, "0000000000")

	if not silent:
		print(len(y), len(sills))
		plt.title("Sillh function for different number of clusters")
		plt.xlabel("number of clusters")
		plt.ylabel("sillh value")
		plt.plot(y, sills)
		plt.show()

		print(sills)

	max_val = max(sills)
	no_offset = sills.index(max_val)
	max_clus = no_offset + y[0]

	if not silent:
		print("best sillh value:", max_val, "\nbest number of clusters:", max_clus)

		print(clusts[no_offset].labels_)
	clusters = create_clusters(clusts[no_offset].labels_)

	if not silent:
		print(clusters)

	return clusts[no_offset]


if __name__ == "__main__":
	find_best(False)
