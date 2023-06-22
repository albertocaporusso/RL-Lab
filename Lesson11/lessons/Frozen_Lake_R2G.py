import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# TODO: implement the following functions as in the previous lessons

class OverrideReward( gymnasium.wrappers.NormalizeReward ):
    """
    Gymansium wrapper useful to update the reward function of the environment

    """
    def step(self, action):

            observation, reward, terminated, truncated, info = self.env.step( action )
            if observation in [5,7,11,12]:
                reward = -5
            elif observation == 15:
                reward = 5
            else:
                reward = 0
                #current_row, current_col = np.unravel_index(observation, (self.env.nrow, self.env.ncol))
                #distance = abs(current_row - 3) + abs(current_col - 3)
                #reward = -0.01*distance
            return observation, reward, terminated, truncated, info

def createDNN( nInputs, nOutputs, nLayer, nNodes ):
    model = Sequential()

    model.add(Dense(nNodes,input_dim = nInputs, activation = 'relu'))
    for _ in range(nLayer - 1):
        model.add(Dense(nNodes, activation = "relu"))
    model.add(Dense(nOutputs, activation = "softmax"))

    return model


def training_loop( env, neural_net, updateRule, frequency=10, episodes=100 ):
    """
    Main loop of the reinforcement learning algorithm. Execute the actions and interact
    with the environment to collect the experience for the training.

    Args:
        env: gymnasium environment for the training
        neural_net: the model to train
        updateRule: external function for the training of the neural network

    Returns:
        averaged_rewards: array with the averaged rewards obtained

    """

    #TODO: initialize the optimizer
    optimizer = tf.keras.optimizers.Adam()
    rewards_list, reward_queue = [], collections.deque( maxlen=100 )
    memory_buffer = []
    count = 0
    n_action = env.action_space.n
    for ep in range(episodes):
        state = env.reset()[0]
        state = np.array(state).reshape(-1)
        ep_reward = 0
        episode = []
        ep_len = 0

        while True:

            #TODO: select the action to perform
            distribution = neural_net(state).numpy()[0]
            action = np.random.choice(n_action, p = distribution)

            #TODO: Perform the action, store the data in the memory buffer and update the reward
            next_state, reward, terminated,truncated,_ = env.step(action)
            if next_state == 15:
                count+=1
            next_state = np.array(next_state).reshape(-1)
            done = terminated or truncated


            episode.append([state, action, next_state, reward, done])
            ep_reward += reward
            ep_len+=1

            #TODO: exit condition for the episode
            if done:
                print(next_state)
                break

            #TODO: update the current state
            state = next_state
        memory_buffer.append(np.asarray(episode))
        #TODO: Perform the actual training every 'frequency' episodes
        if ep % frequency == 0:
            updateRule( neural_net,memory_buffer, optimizer )
            memory_buffer = []


        # Update the reward list to return
        reward_queue.append( ep_reward )
        rewards_list.append( np.mean(reward_queue) )
        print( f"episode {ep:4d}: rw: {ep_reward:3f} (averaged: {np.mean(reward_queue):5.2f})  ep_len = {ep_len}   count: {count}" )

    # Close the enviornment and return the rewards list
    env.close()
    return rewards_list

def REINFORCE_rw2go( neural_net, memory_buffer, optimizer ):
    """
    Main update rule for the REINFORCE process, with the addition of the reward-to-go trick,

    """
    memory_buffer = np.asarray(memory_buffer)
    r = [np.flip(np.cumsum(ep[:,3])) for ep in memory_buffer]
    with tf.GradientTape() as tape:

        objectives= []
        for i,ep in enumerate(memory_buffer):
            state = np.vstack(ep[:,0])
            action = ep[:,1]
            probabilities = neural_net(state)
            indices = tf.transpose(tf.stack([tf.range(probabilities.shape[0]), action]))
            probs = tf.gather_nd(
                indices=indices,
                params=probabilities
            )
            target = tf.math.reduce_sum(tf.math.log(probs)*r[i])
            objectives.append(target)

        objective = -tf.math.reduce_mean(objectives)
        grads = tape.gradient(objective, neural_net.trainable_variables)
        optimizer.apply_gradients(zip(grads, neural_net.trainable_variables))

def main():
    print( "\n***************************************************" )
    print( "*  Welcome to the eleventh lesson of the RL-Lab!  *" )
    print( "*                 (DRL in Practice)               *" )
    print( "***************************************************\n" )

    _training_steps = 10000


    env = gymnasium.make( "FrozenLake-v1",desc=None, map_name="4x4", is_slippery=False )
    env = OverrideReward(env)

    # Create the networks and perform the actual training
    dim_state = 1
    n_action = env.action_space.n
    neural_net = createDNN( dim_state, n_action, nLayer=2, nNodes=32)
    rewards_rw2go = training_loop( env, neural_net, REINFORCE_rw2go,frequency=1, episodes=_training_steps )

    # Plot the resultss
    t = np.arange(0, _training_steps)
    plt.plot(t, rewards_rw2go, label="RW2GO", linewidth=3)
    plt.xlabel( "epsiodes", fontsize=16)
    plt.ylabel( "length", fontsize=16)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()