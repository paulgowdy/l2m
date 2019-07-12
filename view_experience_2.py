import numpy as np
import pickle
import matplotlib.pyplot as plt

with open('saved_experience_normed.p', 'rb') as f:

    exp = pickle.load(f)
    actions = exp[1]
    obs = exp[0]

print(actions.shape)
print(obs.shape)

for i in range(527):

    print(i, np.mean(obs[:,i]), np.std(obs[:,i]))
