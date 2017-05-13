from collections import defaultdict

import numpy as np

n_events = 10  # number of subtopics ver referencia 11
n_tweets = 10 #number of tweets in the event
size_vocab_textual = 100000 #number of textual words
size_vocab_visual = 1000 #number of visual words
#TODO load textual and visual words

lambda_tg = np.full(size_vocab_textual, 0.1)
lambda_ts = np.full(size_vocab_textual, 0.01)
lambda_vg = np.full(size_vocab_visual, 0.1)
lambda_vs = np.full(size_vocab_textual, 1)
beta_z = np.full(n_events, 0.1)  # 0.1
beta_r = np.full(size_vocab_visual, 0.1)
beta_q = np.full(size_vocab_visual, 0.1)

phi_zeta = np.random.dirichlet(beta_z) #Distribucion de probabilidades de los subeventos
phi_tg = np.random.dirichlet(lambda_tg) #Distribución general de palabras, ie, probabilidad de generar una palabra
phi_vg = np.random.dirichlet(lambda_vg) #idem pero visual words

#cada elemento del dict es la distribución de probabilidad de las palabras especificas del sub evento
#Probabilidad que el K produzca la palabra W
phi_ts_dict = defaultdict(list)
phi_vs_dict = defaultdict(list)
for k in range(n_events):
    phi_ts_dict[k] = np.random.dirichlet(lambda_ts)
    phi_vs_dict[k] = np.random.dirichlet(lambda_vs)

zi_dict = defaultdict(list)
phi_r_dict = defaultdict(list)
phi_q_dict = defaultdict(list)
rij_dict = defaultdict(list)
for i in range(n_tweets):
    zi_dict[i] = np.random.multinomial(n_events, phi_zeta)
    phi_r_dict[i] = np.random.dirichlet(beta_r) #Revisar
    phi_q_dict[i] = np.random.dirichlet(beta_q) #Revisar
    tweet = ''
    for j in len(tweet):
        rij_dict[i] = np.random.multinomial(2,phi_r_dict[i])



