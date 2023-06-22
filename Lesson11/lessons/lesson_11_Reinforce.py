import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections


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
    for ep in range(episodes):
        state = env.reset()[0]
        ep_reward = 0
        ep_length = 0
        episode = []
        while True:

            #TODO: select the action to perform
            distribution = neural_net(state.reshape(-1,2)).numpy()[0]
            action = np.random.choice(3, p = distribution)

            #TODO: Perform the action, store the data in the memory buffer and update the reward
            next_state, reward, terminated,truncated,_ = env.step(action)
            next_state = next_state.reshape(-1,2)
            done = terminated or truncated

            episode.append([state, action, next_state, reward, done])
            ep_reward += reward
            ep_length +=1

            #TODO: exit condition for the episode
            if done: break

            #TODO: update the current state
            state = next_state
        memory_buffer.append(np.array(episode))
        #TODO: Perform the actual training every 'frequency' episodes
        if ep % frequency == 0:
            updateRule( neural_net,memory_buffer, optimizer )
            memory_buffer = []


        # Update the reward list to return
        reward_queue.append( ep_reward )
        rewards_list.append( np.mean(reward_queue) )
        print( f"episode {ep:4d}: rw: {int(ep_reward):3d} (averaged: {np.mean(reward_queue):5.2f}),  episode lenght: {ep_length}" )

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
#implement the following class
class OverrideReward( gymnasium.wrappers.NormalizeReward ):
    """
    Gymansium wrapper useful to update the reward function of the environment

    """

    def step(self, action):
            previous_observation = np.array(self.env.state, dtype=np.float32)
            observation, reward, terminated, truncated, info = self.env.step( action )

            position, velocity = observation[0], observation[1]

            if position - previous_observation[0] > 0 and action == 2: reward = 1 #+ abs(0.5 - position)
            if velocity == 0: reward = 0
            if position - previous_observation[0] < 0 and action == 0: reward = 1
            if position >= 0.5: reward = 100


            return observation, reward, terminated, truncated, info


def main():
    print( "\n***************************************************" )
    print( "*  Welcome to the eleventh lesson of the RL-Lab!  *" )
    print( "*                 (DRL in Practice)               *" )
    print( "***************************************************\n" )

    _training_steps = 2500

    # Crete the environment and add the wrapper for the custom reward function
    gymnasium.envs.register(
        id='MountainCarMyVersion-v0',
        entry_point='gymnasium.envs.classic_control:MountainCarEnv',
        max_episode_steps=1000
    )
    env = gymnasium.make( "MountainCarMyVersion-v0" )
    env = OverrideReward(env)

    # Create the networks and perform the actual training
    dim_state = env.observation_space.shape[0]
    n_action = env.action_space.n
    network = createDNN( dim_state, n_action, nLayer=2, nNodes=32)
    rewards_training = training_loop( env, network, REINFORCE_rw2go, frequency=1, episodes=_training_steps  )

    # Save the trained neural network

    # Plot the results
    t = np.arange(0, _training_steps)
    plt.plot(t, rewards_training, label="A2C", linewidth=3)
    plt.xlabel( "epsiodes", fontsize=16)
    plt.ylabel( "length", fontsize=16)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()