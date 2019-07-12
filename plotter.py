import matplotlib.pyplot as plt
import glob

train_logs = glob.glob('train_logs/agent*.txt')

train_durs = []
train_steps = []

plt.figure()
plt.grid()

base_increment = 0.5 / float(len(train_logs))


for tl in train_logs:

    with open(tl, 'r') as f:

        z = f.readlines()

        #z = [float(x.strip()) for x in z]
        l = [x.split(',') for x in z]

        z = [float(x[0]) for x in l]
        durs = [float(x[1]) for x in l]
        steps = [int(x[2].strip()) for x in l]

        train_durs.append(durs)
        train_steps.append(steps)

        #print(durs)
        #print(steps)
        print(len(z))
        #print('')

        plt.plot(z, c = (base_increment * train_logs.index(tl), 0.1, 0.6, 0.1))
        plt.title('Training Ep Reward')
        plt.xlabel('Training Ep')

'''
plt.figure()
plt.grid()
plt.title('Total Ep Duration')

for agent in train_durs:

    plt.plot(agent, c = (base_increment * train_durs.index(agent), 0.2, 0.5, 0.3))
'''

plt.figure()
plt.grid()
plt.title('Train Ep Seconds Per Step')

for i in range(len(train_durs)):

    s_per_step = [dur/steps for dur, steps in zip(train_durs[i], train_steps[i])]

    #print(train_durs[i], train_steps[i])

    plt.plot(s_per_step, c = (base_increment * i, 0.2, 0.5, 0.3))



plt.show()
