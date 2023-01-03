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


# ===============================================================================================
# ===============================================================================================
# == Clustering ==


def similarity(r1, r2):
    # |intersection r1,r2|*2 / (|r1|+|r2|)
	global rules

	intersect = rules[r1].intersection(rules[r2])
	return 2 * len(intersect) / (len(rules[r1]) + len(rules[r2]))


def dissimilarity(r1, r2):
	return 1 - similarity(r1, r2)


# ===============================================================================================
# ===============================================================================================
# == Sillh ==

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


def get_neighbor(i, n_clusters):
    # should change to reflect the distances.
	if i == n_clusters - 1 or not i % 2:
		return i - 1
	return i + 1


# ===============================================================================================
# ===============================================================================================


def a_r(r):
	cluster = labels[r]
	nomin = 0
	for i in clusters[cluster]:
		if i == r:
			continue
		nomin += dist_matr[i][r]
	denom = len(clusters[cluster]) - 1

	return nomin / denom


def b_r(r):
	cluster = labels[get_neighbor(r)]
	nomin = 0
	for i in clusters[cluster]:
		nomin += dist_matr[i][r]
	denom = len(clusters[cluster])
	return nomin / denom


def sillh(r):
	res = 0
	for i in range(rules_num):
		b = b_r(r)
		a = a_r(r)
		res += (b - a) / max(b, a)
	res /= rules_num
	return res


# ==============================================================================================
# ==============================================================================================
# == CP by Quality ==

def initial_proxies():
	proxies = list()
	for clus in range(n_clusters):
		max_qual = 0
		max_i = -1
		for i in clusters[clus]:
			qual = info_rules.at[i, 'QUALITY']
			if qual > max_qual:
				max_qual = qual
				max_i = i
		proxies.append(max_i)
	return proxies


# ==============================================================================================
# ==============================================================================================
# == Improve Proxy Rules == 

def cov_rate(x, U):
	"""
    :param x: any instance
    :param U: a subset of the complete set of rules
    :return: a value between 0 and 1 that is coverage rate of x and U
	"""
	counter = 0
	for r in U:
		if x in rules[r]:
			counter += 1
	return counter / len(U)


def adj_cov_rate(x, U, R):
	"""
    :param x: an instance
    :param U: a subset of the complete set of rules
    :param R: complete set of rules
    :return: Round(covRate(x, R) . |U|) / |U|
	"""
	return round(cov_rate(x, R) * len(U)) / len(U)


def representativeness(R_eta, R, U):
	"""
    :param R_eta: set of cluster proxy rules R_eta over the clustering eta
    :param R: a set of induced rules
    :param U: a subset of complete set of rules
    :return: representativeness(R_eta, R)
	"""
	counter = 0
    # todo: ids is complete set of instances
	for instance in ids:
		counter += abs(adj_cov_rate(instance, U, R) - cov_rate(instance, R_eta))
	return 1 - counter / len(ids)


# == Genetic algorithm == 

def create_genome():
	return [random.choice(clusters[clus]) for clus in range(n_clusters)]


def mate(parent1, parent2):
	child = ""
	for i in range(n_clusters):
		prob = random.random()
		if prob < 0.45:
			child.append(parent1[i])
		elif prob < 0.9:
			child.append(parent2[i])
		else:
			child.append(random.choice(clusters[i]))

	return child


def genetic_algo():
	population_size = 20
	iteration_num = 100

	best_repr = 0
	best_genome = ""

	population = [create_genome() for i in range(population_size)]
	for ite in range(iteration_num):
		population = sorted(population, key=lambda x: representativeness(x), reverse=True)

		if representativeness(population[0]) > best_repr:
			best_repr = representativeness(population[0])
			best_genome = population[0]

        # directly go to next
		choose = (10 * population_size) // 100
		new_generation = population[0:choose]

		while len(new_generation) != population_size:
			parent1 = random.choice(population[:pop_cut])
			parent2 = random.choice(population[:pop_cut])
			child = mate(parent1, parent2)
			new_generation.append(child)

		population = new_generation

	return best_genome


def choose_best_proxies():
	best_gen = genetic_algo()
	best_prox = initial_proxies()

	if representativeness(best_gen) > representativeness(best_prox):
		return best_gen
	
	return best_prox


#================================================================================================
#================================================================================================


def read_rules(file_name='rules.pkl'):
	"""
     Return exactly the pysubgroup output (a dataframe contain quality, subgroup, ...)
	"""
	return pd.read_pickle("pickle_data/" + file_name)


def reed_id_set(file_name='id_set.pkl'):
	"""
        Return a dictionary {rule_ids:set_of_instances}
	"""
	with open('pickle_data/' + file_name, 'rb') as handle:
		return pickle.load(handle)


# ===============================================================================================
# ===============================================================================================
# == Main funcs == 

if __name__ == "__main__":
	rules = dict()

	rules[0] = {5, 6, 7, 4}
	rules[1] = {1, 2, 4, 3}
	rules[2] = {1, 5, 7, 3, 4}

	rules_num = len(rules)
	print(len(rules))
	dissimilarity(1, 2)

	dist_matr = np.array([[dissimilarity(i, j) for i in range(rules_num)] for j in range(rules_num)])
	cond = squareform(dist_matr)
	print(dist_matr)
	print(cond)

    # clustering = aglo(n_clusters=2,linkage='average').fit(dist_matr)




	clustering = aglo(n_clusters=None, distance_threshold=0, linkage='average').fit(dist_matr)
	plot_dendogram(clustering)
	plt.show()


	n_clusters = 2
	clustering = aglo(distance_threshold=None, linkage='average', n_clusters=n_clusters).fit(dist_matr)

	labels = clustering.labels_
	clusters = defaultdict(list)
	for i, clus in enumerate(labels):
		clusters[clus].append(i)

	print(clusters)

	# plot_dendogram(clustering)
	# plt.show()
# print(clustering)

	# Z = linkage(dist_matr,method='average')
# print(Z)

	# dendrogram(Z)
	# plt.show()






# plot_dendogram(clustering)
# plt.show()
# it gets n clusters. then I should calc sillhoute. =)

# 	from sklearn.cluster import AgglomerativeClustering 
# from sklearn.datasets import make_blobs
# from scipy.spatial import distance_matrix
# from scipy.cluster import hierarchy
# from scipy.spatial.distance import squareform

# X1, y1 = make_blobs(n_samples=50, centers=[[4,4],
#                                            [-2, -1],
#                                            [1, 1],
#                                            [10,4]], cluster_std=0.9)

# agglom = AgglomerativeClustering(n_clusters = 4, linkage = 'average')
# agglom.fit(X1,y1)

# dist_matrix = distance_matrix(X1,X1)
# print(dist_matrix.shape)
# condensed_dist_matrix = squareform(dist_matrix)
# print(condensed_dist_matrix.shape)
# Z = hierarchy.linkage(condensed_dist_matrix, 'complete')
