import pickle
from sklearn import cluster
import numpy as np
from numpy import linalg as LA


def get_images_subevent(weight_matrix, subevent):
    images = np.array([np.zeros(len(subevent)) for i in range(len(subevent))])
    for i in range(len(subevent)):
        for j in range(i):
            index_i = subevent[i]
            index_j = subevent[j]
            images[i][j] = weight_matrix[index_i][index_j]
    return images


def mainfold_ranking(cluster_elements):
    h = np.array(np.ones(len(cluster_elements)))  # TODO cambiar este arreglo, por la distribución de probabilidades
    gamma = 0.85

    diag = np.diag(np.diag(cluster_elements))
    for i in range(diag.shape[0]):
        for j in range(diag.shape[1]):
            if i == j and diag[i][j] != 0:
                diag[i][j] = 1 / np.sqrt(diag[i][j])

    second_term = (1 - gamma) * h
    first_term = gamma * np.dot(diag, np.dot(cluster_elements, diag))
    ranking_vector_new = h
    convergence = (1 - gamma) * np.dot(LA.inv(np.identity(cluster_elements.shape[0]) - first_term), h)
    while (ranking_vector_new < convergence).all():
        ranking_vector_old = ranking_vector_new
        ranking_vector_new = np.dot(first_term, ranking_vector_old) + second_term

    return ranking_vector_new


subevent = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            17]  # TODO cargar conjunto de images correspondientes al subevento
weight_matrix = pickle.load(open('weight_matrix.p', 'rb'))
images_matrix = get_images_subevent(weight_matrix, subevent)
spectral = cluster.SpectralClustering(eigen_solver='arpack', affinity="nearest_neighbors")
labels = spectral.fit(images_matrix).labels_
ranking_cluster = []
selected = []
for i in range(spectral.n_clusters):
    indexes = [index for index, value in enumerate(labels) if value == i]
    cluster_elements = []
    for j in range(len(indexes)):
        cluster_elements.append(images_matrix[j])
    ranking_cluster.append(mainfold_ranking(images_matrix))
    #selected.append(mainfold_ranking(images_matrix))
for list_ranking in ranking_cluster:
    selected.append(list_ranking.indexes(max(list_ranking)))

print(list_ranking)
