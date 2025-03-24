import numpy as np
import pickle
import simple_custom_taxi_env as taxi_env

def sqrt(x):
    if x == 0:
       return 0
    if x == 1:
       return 1
    if x >= 2 and x <= 4:
       return 2
    if x >= 5:
       return 3
    if x == -1:
       return -1
    if x <= -2 and x >= -4:
       return -2
    if x <= -5:
       return -3

def extract_state(state):
    taxi_row, taxi_col = state[0], state[1]
    station0_row, station0_col = state[2], state[3]
    station1_row, station1_col = state[4], state[5]
    station2_row, station2_col = state[6], state[7]
    station3_row, station3_col = state[8], state[9]
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = state[10], state[11], state[12], state[13], state[14], state[15]

    relative0_row = station0_row - taxi_row
    relative0_col = station0_col - taxi_col
    dis0 = abs(relative0_row) + abs(relative0_col)
    if dis0 >= 3:
       dis0 = (dis0 + 1) // 2
    relative1_row = station1_row - taxi_row
    relative1_col = station1_col - taxi_col
    dis1 = abs(relative1_row) + abs(relative1_col)
    if dis1 >= 3:
       dis1 = (dis1 + 1) // 2
    relative2_row = station2_row - taxi_row
    relative2_col = station2_col - taxi_col
    dis2 = abs(relative2_row) + abs(relative2_col)
    if dis2 >= 3:
       dis2 = (dis2 + 1) // 2
    relative3_row = station3_row - taxi_row
    relative3_col = station3_col - taxi_col
    dis3 = abs(relative3_row) + abs(relative3_col)
    if dis3 >= 3:
       dis3 = (dis3 + 1) // 2
            
    # v0
    # return (taxi_row, taxi_col, obstacle_north, obstacle_south, obstacle_east, obstacle_west)
    # v1
    # return (relative0_row,relative0_col,relative1_row,relative1_col,relative2_row,relative2_col,relative3_row,relative3_col,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
    # v2
    return (dis0, dis1, dis2 ,dis3 ,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)

def q_table_learning(episodes,alpha,gamma,epsilon_start,epsilon_end,decay_rate,env_config):
    q_table = {}
    state_count = 0

    rewards_per_episode = []
    epsilon = epsilon_start

    for episode in range(episodes):
        env = taxi_env.SimpleTaxiEnv(**env_config)
        obs, _ = env.reset()

        state = extract_state(obs)
        done, truncated = False, False

        total_reward = 0
        episode_step = 0

        while not done and not truncated:
            # TODO: Initialize the state in the Q-table if not already present.
            if state not in q_table:
              q_table[state] = np.zeros((6))
              state_count += 1

            # TODO: Implement ε-greedy policy for action selection.
            if np.random.uniform(0,1) < epsilon:
              action = np.random.randint(0,5)
            else:
              action = np.argmax(q_table[state])

            # Execute the selected action.
            obs, reward, done, _ = env.step(action)
            next_state = extract_state(obs)
            on_station = next_state[0] == 0 or next_state[1] == 0 or next_state[2] == 0 or next_state[3] == 0
            next_on_station = next_state[0] == 0 or next_state[1] == 0 or next_state[2] == 0 or next_state[3] == 0
            if state[8] == 0 or state[9] == 0 and state[:4] == next_state[:4]:
              reward -= 4
            if next_on_station:
              reward += 1
            if state[8] and not next_state[8]:
              reward -= 5
            if next_on_station and (not state[8] and not state[9]):
              reward -= 5
            if next_on_station and (next_state[9] and not next_state[8]):
              reward -= 5
            if on_station and state[8] and not state[9]:
              reward -= 2
            if next_on_station and (next_state[9] and next_state[8]):
              reward += 3
            if on_station and next_state[9] and next_state[8] and state[:4] == next_state[:4]:
              reward += 5
               
            total_reward += reward
            if next_state not in q_table:
              q_table[next_state] = np.zeros((6))
              state_count += 1
            
            best_action = np.argmax(q_table[next_state])
            q_table[state][action] += alpha * (reward + gamma * q_table[next_state][best_action] - q_table[state][action])

            state = next_state
            episode_step += 1

        rewards_per_episode.append(total_reward)
        epsilon = max(epsilon_end, epsilon * decay_rate)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.4f}, Epsilon: {epsilon:.3f}")
            print(f"state_count {state_count}")
    return q_table
if __name__ == "__main__":
  env_config = {
      "fuel_limit": 5000,
      "grid_size": 10
  }
  q_table = q_table_learning(20000,0.1,0.99,1.0,0.1,0.9999,env_config)
  with open('./q_table.pkl', 'wb') as f:
      pickle.dump(q_table, f, protocol=pickle.HIGHEST_PROTOCOL)
