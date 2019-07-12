import numpy as np
import pickle
import matplotlib.pyplot as plt

with open('saved_experience.p', 'rb') as f:

    exp = pickle.load(f)
    actions = exp[1]
    obs = exp[0]

print(actions.shape)
print(obs.shape)

mean_collect = [0.0] * 242
std_collect = [1.0] * 242


for i in range(242,527):

    mean = np.mean(obs[:,i])
    std = np.std(obs[:,i])

    mean_collect.append(mean)

    if std > 0.1:

        std_collect.append(std)

    else:

        std_collect.append(1.0)

print(len(mean_collect))
print(len(std_collect))

plt.figure()
plt.plot(mean_collect)

plt.figure()
plt.plot(std_collect)

plt.show()

with open('norm_sample.p', 'wb') as f:

    pickle.dump([mean_collect, std_collect], f)
