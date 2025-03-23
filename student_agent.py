# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import train

with open("./q_table.pkl", "rb") as f:
    Q_table = pickle.load(f)

def cap(x):
    if x > 0:
        return min(x,5)
    elif x < 0:
        return max(x,-5)
    else:
        return 0

def get_action(obs):

    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    obs = train.extract_state(obs)
    if obs not in Q_table:
        tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10 ,tmp11, tmp12, tmp13 = obs
        tmp0 = cap(tmp0)
        tmp1 = cap(tmp1)
        tmp2 = cap(tmp2)
        tmp3 = cap(tmp3)
        tmp4 = cap(tmp4)
        tmp5 = cap(tmp5)
        tmp6 = cap(tmp6)
        tmp7 = cap(tmp7)
        obs = tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10 ,tmp11, tmp12, tmp13
        if obs not in Q_table:
            return random.randint(0,5)

    return np.argmax(Q_table[obs])
    # You can submit this random agent to evaluate the performance of a purely random strategy.

