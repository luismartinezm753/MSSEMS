import numpy as np

K = 10  # number of subtopics ver referencia 11
n_tweets = 10 #number of tweets in the event
size_vocab_textual = 100000 #number of textual words
size_vocab_visual = 1000 #number of visual words
#TODO load textual and visual words

lambda_tg = np.full(size_vocab_textual,0.1)
lambda_ts = np.full(size_vocab_textual,0.01)
lambda_vg = np.full(size_vocab_visual,0.1)
lambda_vs = np.full(size_vocab_textual,1)
beta_z = np.full(K,0.1) #0.1
beta_r = np.full(size_vocab_visual,0.1)
beta_q = np.full(size_vocab_visual,0.1)


dist_tg = np.random.dirichlet(lambda_tg)
dist_vg = np.random.dirichlet(lambda_vg)

fi_zeta = np.random.dirichlet(beta_z)

specific_textual = []
specific_visual = []
zeta = []
fi_r = []
fi_q = []

for k in range(K):
    specific_textual.append(np.random.dirichlet(lambda_ts))
    specific_visual.append(np.random.dirichlet(lambda_vs))

for j in range(n_tweets):
    zeta.append(np.random.multinomial(1,beta_z))
    fi_r.append(np.random.dirichlet(beta_r))
    fi_q.append(np.random.dirichlet(beta_q))
