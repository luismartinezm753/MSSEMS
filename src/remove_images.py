import itertools
import pickle
from math import exp
from numpy import linalg as LA
from scipy.sparse import csgraph
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
from matplotlib import pyplot as plt

from Bag import *

bandwith = 0.1
k = 20
m = 40
alfa_1 = 0.1
alfa_2 = 0.1
zeta = 0.1
threshold = 0.4


# K en el paper es 20, bandwith es 0.1, m=20, alfa_1=0.1, alfa_2=0.1

def calculate_W(distances, n_images, indices):
    W = np.array([np.zeros(n_images) for i in range(n_images)])
    for i in range(n_images):
        for j in range(k):
            W[i][indices[i][j]] = exp(-(pow(distances[i][j], 2) / pow(bandwith, 2)))
    return W


def function_j(y, a, diagonal, eigenvector):
    first_term = pow(LA.norm(np.dot(eigenvector, a) - y), 2)
    second_term = alfa_2 * np.dot(np.dot(np.transpose(a), diagonal), a)
    return first_term + second_term


def function_delta_j(y, a, diagonal, eigenvector):
    first_term = 2 * np.dot((np.dot(np.transpose(eigenvector), eigenvector) + alfa_2 * diagonal), a)
    second_term = 2 * np.dot(np.transpose(eigenvector), y)
    return first_term - second_term


# Esta funcion deberia resolver el problema de minimizacion explicado en la EQ 1 del papper ver referencia 28
def spectral_filter(relevance, eigenvectors, diag_eigenvalues):
    beta = 0.5
    n = 0.01
    epsilon = 0.01
    y_new = relevance
    a_j = []
    count_iter = 0
    while count_iter<15:
        y_old = y_new
        a_old = (LA.inv(np.dot(np.transpose(eigenvectors), eigenvectors) + alfa_2 * diag_eigenvalues).dot(
            np.transpose(eigenvectors))).dot(y_old)  # es a(t)
        for j in itertools.count():
            beta_new = pow(beta, j) #TODO calcular correctamente beta
            a_new = ball_projection(a_old - beta_new * function_delta_j(y_old, a_old, diag_eigenvalues, eigenvectors))# es a(t+1)
            if (function_j(y_old, a_new, diag_eigenvalues, eigenvectors) - function_j(y_old, a_old, diag_eigenvalues,eigenvectors)) < epsilon:
                a_j = a_new
                a_old = a_new
                break
            a_old = a_new
        y_new = round(eigenvectors.dot(a_j))
        count_iter = count_iter + 1
        print(count_iter)
        # if LA.norm(y_new) - LA.norm(y_old) < epsilon:
        #     break
    return y_new


def ball_projection(vector):
    if sum(vector) < zeta:
        return vector
    else:
        v = np.sort(np.absolute(vector))
        iter = (v[j] - ((1 / j) * sum([v[j_2] - zeta for j_2 in range(j)])) for j in range(1,len(vector)))
        r_list = np.fromiter(iter, np.float)
        iter = (k for k in range(len(r_list)) if r_list[k] > 0)
        r = np.max(np.fromiter(iter, np.float))
        theta = (1 / r) * sum([v[j] - zeta for j in range(int(r))])
        iter_a = (np.sign(vector[j]) * np.maximum(abs(vector[j] - theta), 0) for j in range(len(vector)))
        return np.fromiter(iter_a, np.float, count=len(vector))


def round(a):
    max = np.max(a)
    y = np.array(np.zeros(len(a)))
    for i in range(len(a)):
        if a[i] > max * threshold:
            y[i] = 1
    return y


bov = BOV(no_clusters=20)
bov.train_path = "/home/luism/Universidad/images/"
filter_path="/home/luism/Universidad/images_filter/"
histogram_images_normalized_loc = Path('histogram_images_normalized.p')

if histogram_images_normalized_loc.exists():
    with histogram_images_normalized_loc.open('rb') as f:
        histogram_images_normalized = pickle.loads(f.read())
        bov.file_helper.n_images = histogram_images_normalized.shape[0]
else:
    histogram_images = bov.trainModel()
    histogram_images_normalized = preprocessing.normalize(histogram_images, norm='l2')
    pickle.dump(histogram_images_normalized, open("histogram_images_normalized.p", "wb"))


print("Calculando distancias")
nbrs = NearestNeighbors(n_neighbors=k).fit(histogram_images_normalized)
distances, indices = nbrs.kneighbors(histogram_images_normalized)
weight_matrix = calculate_W(distances=distances, indices=indices, n_images=bov.file_helper.n_images)
pickle.dump(weight_matrix,open( "weight_matrix.p", "wb" ))
laplacian_graph = csgraph.laplacian(weight_matrix, normed=False)

print("Calculando Eigenvalues")
eigenvalues, eigenvector = LA.eigh(laplacian_graph)
relevance = np.array(np.ones(bov.file_helper.n_images))
smooth_eigenvectors = eigenvector[1:m + 1]  # Representa U en el papper
diag_eigenvalues = np.diag(eigenvalues[1:m + 1])

print("Aplicando filtro espectral")
denoise_vector = spectral_filter(relevance=relevance, eigenvectors=np.transpose(smooth_eigenvectors),
                                 diag_eigenvalues=diag_eigenvalues)
index_images = [j for j in range(len(denoise_vector)) if denoise_vector[j] > 0]
im_list, count = bov.file_helper.getFiles(bov.train_path)
print(index_images)
for importan in index_images:
    plt.imshow(im_list['baltimore'][importan], interpolation='nearest')
    plt.show()
