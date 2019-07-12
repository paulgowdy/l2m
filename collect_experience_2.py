from osim.env import L2M2019Env
from osim.control.osim_loco_reflex_song2019 import OsimReflexCtrl
import numpy as np
import pickle

mode = '2D'
difficulty = 1
visualize=False
seed=None
sim_dt = 0.01
sim_t = 5
timstep_limit = int(round(sim_t/sim_dt))


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

if mode is '2D':
    params = np.loadtxt('params2d.txt')
elif mode is '3D':
    params = np.loadtxt('params_3D_init.txt')





locoCtrl = OsimReflexCtrl(mode=mode, dt=sim_dt)

control_env = L2M2019Env(visualize=visualize, seed=seed, difficulty=difficulty)
control_env.change_model(model=mode, difficulty=difficulty, seed=seed)
obs_dict_action = control_env.reset(project=True, seed=seed, obs_as_dict=True, init_pose=INIT_POSE)
control_env.spec.timestep_limit = timstep_limit

obs_env = L2M2019Env(visualize=False, seed=seed, difficulty=difficulty)
obs_env.change_model(model=mode, difficulty=difficulty, seed=seed)
obs_dict_record = obs_env.reset(project=False, seed=seed, obs_as_dict=False, init_pose=INIT_POSE)
obs_env.spec.timestep_limit = timstep_limit

with open('norm_sample.p', 'rb') as f:

    norm_sample = pickle.load(f)

    means = norm_sample[0]
    stds = norm_sample[1]


def process_obs_dict(obs_dict):

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





total_reward = 0
t = 0
i = 0

obs_collect = []
action_collect = []

while True:
    i += 1
    t += sim_dt

    proc_obs = process_obs_dict(obs_dict_record)

    #print(proc_obs)

    locoCtrl.set_control_params(params)

    action = locoCtrl.update(obs_dict_action)

    obs_collect.append(proc_obs)
    action_collect.append(action)

    obs_dict_action, reward, done, info = control_env.step(action, project = True, obs_as_dict=True)
    obs_dict_record, reward_obs, done_obs, info_obs = obs_env.step(action, project = False, obs_as_dict=False)

    print(i, reward)
    #print(action)
    #print(len(obs_dict_record))

    print('')
    total_reward += reward

    if done:
        break

print('    score={} time={}sec'.format(total_reward, t))

obs_collect = np.array(obs_collect)
action_collect = np.array(action_collect)

print(obs_collect.shape)
print(action_collect.shape)

with open('saved_experience_normed.p', 'wb') as f:

    pickle.dump([obs_collect, action_collect], f)
