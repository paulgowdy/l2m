'''
## Env Wrapper ##
# A wrapper around the OpenAI Gym environments. Each environment requires its own custom wrapper as the preprocessing required differs by environment.
@author: Mark Sinton (msinto93@gmail.com)
'''

#import gym
from osim.env import L2M2019Env
from params import train_params

import numpy as np
from collections import deque
import pickle

'''
def proc_obs_dict(obs_dict):

    new_obs = []

    pelvis_absolute_pos = obs_dict['body_pos']['pelvis']

    new_obs.extend(pelvis_absolute_pos)

    for k in obs_dict['body_pos'].keys():

        if k != 'pelvis':

            new_obs.extend([a - b for a, b in zip(obs_dict['body_pos'][k], pelvis_absolute_pos)])

    new_obs.extend([a - b for a, b in zip(obs_dict['misc']['mass_center_pos'], pelvis_absolute_pos)])
    new_obs.extend(obs_dict['misc']['mass_center_vel'])
    new_obs.extend(obs_dict['misc']['mass_center_acc'])

    other_keys = ['joint_pos', 'joint_vel', 'body_vel', 'body_acc']


    # Excluded!
    # joint_acc
    # body_pos_rot, body_vel_rot, body_acc_rot
    # muscles, forces


    for k in other_keys:

        for sk in obs_dict[k].keys():

            new_obs.extend(obs_dict[k][sk])

    # now want to grab just the part of the velocity field that's relevant

    pos_x_ind, pos_z_ind = pos_selector(pelvis_absolute_pos[0], pelvis_absolute_pos[2])

    # this runs the risk of wrapping and return the wrong subarray if the agent is on the edge of the field, but thats ok...
    vel_tgt_sub_select = obs_dict['v_tgt_field'][:, pos_x_ind - 1 : pos_x_ind + 2, pos_z_ind - 1 : pos_z_ind + 2]

    l = list(vel_tgt_sub_select.flatten())

    new_obs.extend(l)

    return new_obs
'''
with open('norm_sample.p', 'rb') as f:

    norm_sample = pickle.load(f)

    means = norm_sample[0]
    stds = norm_sample[1]

def proc_obs_dict(obs_dict):

    '''
    for k in obs_dict.keys():

        print(k)

    joint_pos
    joint_vel
    joint_acc
    body_pos
    body_vel
    body_acc
    body_pos_rot
    body_vel_rot
    body_acc_rot
    forces
    muscles
    markers
    misc
    v_tgt_field

    '''

    #print(obs_dict['joint_pos']['knee_r'])
    #print(obs_dict['joint_pos']['knee_l'])
    #print(obs_dict['joint_pos'])
    #print('')


    v_tgt  = obs_dict['v_tgt_field']
    #print(v_tgt.shape) 2,11,11
    v_tgt = v_tgt.flatten() / 10.0


    new_obs = list(v_tgt)

    pelvis_pos = obs_dict['body_pos']['pelvis']

    new_obs.extend(pelvis_pos)

    for k in obs_dict['body_pos'].keys():

        if k != 'pelvis':

            #print(obs_dict['body_pos'][k])
            #print([a - b for a, b in zip(obs_dict['body_pos'][k], pelvis_pos)])
            #print('')

            new_obs.extend([a - b for a, b in zip(obs_dict['body_pos'][k], pelvis_pos)])

    #'muscles', 'misc'
    # , 'forces'

    for k in ['joint_pos', 'joint_vel', 'joint_acc', 'body_vel', 'body_acc', 'body_pos_rot', 'body_vel_rot', 'body_acc_rot']:

        for sub_k in obs_dict[k].keys():

            new_obs.extend(obs_dict[k][sub_k])

    new_obs = [a - b for a,b in zip(new_obs, means)]
    new_obs = [float(a)/float(b) for a,b in zip( new_obs, stds)]



    return new_obs



class L2MWrapper:
    def __init__(self, env_name, vis, mode = '3D', difficulty = 1):
        self.env_name = env_name
        self.env = L2M2019Env(visualize=vis)
        self.env.change_model(model=mode, difficulty=difficulty)
        
        print('setting env accuracy:', train_params.ENV_ACC)
        self.env.osim_model.set_integrator_accuracy(train_params.ENV_ACC)


        self.state_storage = deque([])
        self.state_storage_limit = 1000

    def reset(self):
        '''
        INIT_POSE = np.array([
            1.699999999999999956e+00, # forward speed
            .5, # rightward speed
            9.023245653983965608e-01, # pelvis height
            2.012303881285582852e-01, # trunk lean
            0*np.pi/180, # [right] hip adduct
            -6.952390849304798115e-01, # hip flex
            -3.231075259785813891e-01, # knee extend
            1.709011708233401095e-01, # ankle flex
            0*np.pi/180, # [left] hip adduct
            -5.282323914341899296e-02, # hip flex
            -8.041966456860847323e-01, # knee extend
            -1.745329251994329478e-01]) # ankle flex
        '''
        #state = self.env.reset(project = False, init_pose=INIT_POSE)
        state = self.env.reset(project = False)

        state = proc_obs_dict(state)

        return state

    def get_random_action(self):
        action = self.env.action_space.sample()
        return action
    '''
    def shape_reward(self, obs_dict, reward, done):

        p_x = obs_dict['body_pos']['pelvis'][0]
        h_x = obs_dict['body_pos']['head'][0]

        l_knee_angle = obs_dict['joint_pos']['knee_l'][0]
        r_knee_angle = obs_dict['joint_pos']['knee_r'][0]


        lean_penalty = 0 #10.0 * min(0.3, max(0, p_x - h_x - 0.3))
        left_knee_bend_penalty =  max(0, l_knee_angle + 0.1)# - 0.1)
        right_knee_bend_penalty = max(0, r_knee_angle + 0.1)# - 0.1)

        term_penalty = 0

        if done:

            term_penalty = 1

        #a_list = list(action)

        #muscle_activation_penalty = 0.1 * sum([i ** 2 for i in a_list])

        r = reward - lean_penalty - left_knee_bend_penalty - right_knee_bend_penalty - term_penalty# - muscle_activation_penalty
        #print(reward, r)
        return r



    def shape_reward(self, obs_dict):

        target_values = np.array([
        -0.20123, #ground_pelvis 0
        0.9023,
        0.695,
        -0.323,
        -0.1709,
        0.05282,
        -0.804,
        0.1745
        ])

        obs_values = np.array([

        obs_dict['joint_pos']['ground_pelvis'][0],
        obs_dict['joint_pos']['ground_pelvis'][4],

        obs_dict['joint_pos']['hip_r'][0],
        obs_dict['joint_pos']['knee_r'][0],
        obs_dict['joint_pos']['ankle_r'][0],

        obs_dict['joint_pos']['hip_l'][0],
        obs_dict['joint_pos']['knee_l'][0],
        obs_dict['joint_pos']['ankle_l'][0]
        ])

        squared_error = np.sum(np.square(target_values - obs_values))

        return 0.1 - 0.1 * squared_error

    '''


    def step(self, action):
        #next_state, reward, terminal, _ = self.env.step(action, project = True)

        done = False

        fs_reward = 0

        for i in range(4):
            if not done:
                observation_dict, reward, done, info = self.env.step(action, project = False)

                #reward = self.shape_reward(observation_dict, reward, done)

                fs_reward += reward

        #observation_dict, reward, done, info = self.env.step(action, project = False)

        observation = proc_obs_dict(observation_dict)

        #reward = self.shape_reward(observation_dict)

        #print('obs', len(observation))
        #sub_obs = observation[:-18]
        #sub_obs.extend([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        #print('sub_obs', len(sub_obs))

        #self.state_storage.append(sub_obs)

        #if len(self.state_storage) > self.state_storage_limit:

        #    self.state_storage.popleft()

        #return next_state, reward, terminal
        return observation, reward, done

    def set_random_seed(self, seed):
        self.env.seed(seed)

    def render(self):
        frame = self.env.render(mode='rgb_array')
        return frame

    def close(self):
        self.env.close()

    def normalise_state(self, state):
        # Normalise state values to [-1, 1] range
        #return np.array(state)/train_params.STATE_BOUND_HIGH
        '''
        if len(self.state_storage) > 200:

            # don't want to norm the velocity compenet!

            state_means = np.mean(np.array(list(self.state_storage)), axis = 0)
            #print('state_means shape:', state_means.shape)


            normed_state = np.array(state) - state_means

            # vars? - sometimes they're zero, and other times the var is reaaaally small...

            #print(normed_state)

            return np.array(normed_state)

        else:
        '''
        return np.array(state)

    def normalise_reward(self, reward):
        # Normalise reward values
        return reward

'''
class EnvWrapper:
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = L2M2019Env(visualize=False)

    def reset(self):
        state = self.env.reset(project = False)

        state = proc_obs_dict(state)

        return state

    def get_random_action(self):
        action = self.env.action_space.sample()
        return action

    def step(self, action):
        #next_state, reward, terminal, _ = self.env.step(action, project = True)
        observation, reward, done, info = self.env.step(action, project = False)

        observation = proc_obs_dict(observation)
        #return next_state, reward, terminal
        return observation, reward, done

    def set_random_seed(self, seed):
        self.env.seed(seed)

    def render(self):
        frame = self.env.render(mode='rgb_array')
        return frame

    def close(self):
        self.env.close()

class L2MWrapper(EnvWrapper):

    def __init__(self, env_name):

        EnvWrapper.__init__(self, env_name)

    def normalise_state(self, state):
        # Normalise state values to [-1, 1] range
        return np.array(state)/train_params.STATE_BOUND_HIGH

    def normalise_reward(self, reward):
        # Normalise reward values
        return reward/1.0
'''
