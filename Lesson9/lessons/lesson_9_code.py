import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


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
    dim_state = env.observation_space.shape[0]
    n_action = env.action_space.n
    for ep in range(episodes):
        state = env.reset()[0]
        state = state.reshape(-1,dim_state)
        ep_reward = 0
        episode = []
        while True:

            #TODO: select the action to perform
            distribution = neural_net(state).numpy()[0]
            action = np.random.choice(n_action, p = distribution)

            #TODO: Perform the action, store the data in the memory buffer and update the reward
            next_state, reward, terminated,truncated,_ = env.step(action)
            next_state = next_state.reshape(-1,dim_state)
            done = terminated or truncated

            episode.append([state, action, next_state, reward, done])
            ep_reward += reward

            #TODO: exit condition for the episode
            if done: break

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
        print( f"episode {ep:4d}: rw: {int(ep_reward):3d} (averaged: {np.mean(reward_queue):5.2f})" )

    # Close the enviornment and return the rewards list
    env.close()
    return rewards_list



def REINFORCE_naive( neural_net, memory_buffer, optimizer ):
    """
    Main update rule for the REINFORCE process, the naive implementation of the policy-gradient theorem.

    """

    with tf.GradientTape() as tape:
        memory_buffer = np.asarray(memory_buffer)
        objectives= []
        for ep in memory_buffer:
            state = np.vstack(ep[:,0])
            action = ep[:,1]
            reward = sum(ep[:,3])
            probabilities = neural_net(state)
            indices = tf.transpose(tf.stack([tf.range(probabilities.shape[0]), action]))
            probs = tf.gather_nd(
                indices=indices,
                params=probabilities
            )
            target = tf.math.reduce_sum(tf.math.log(probs))
            objectives.append(target * reward)

        objective = -tf.math.reduce_mean(objectives)
        grads = tape.gradient(objective, neural_net.trainable_variables)
        optimizer.apply_gradients(zip(grads, neural_net.trainable_variables))


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
    print( "\n*************************************************" )
    print( "*  Welcome to the ninth lesson of the RL-Lab!   *" )
    print( "*                 (REINFORCE)                   *" )
    print( "*************************************************\n" )

    _training_steps = 1500
    env = gymnasium.make( "CartPole-v1" )
    dim_state = env.observation_space.shape[0]
    n_action = env.action_space.n
    neural_net = createDNN( dim_state, n_action, nLayer=2, nNodes=32)
    rewards_rw2go = training_loop( env, neural_net, REINFORCE_rw2go,frequency=10, episodes=_training_steps  )

    # Training A)
    neural_net = createDNN( dim_state, n_action, nLayer=2, nNodes=32)
    rewards_naive = training_loop( env, neural_net, REINFORCE_naive,frequency=10, episodes=_training_steps  )


    # Training B)

    # Plot
    t = np.arange(0, _training_steps)
    plt.plot(t, rewards_naive, label="naive", linewidth=3)
    plt.plot(t, rewards_rw2go, label="reward to go", linewidth=3)
    plt.xlabel( "epsiodes", fontsize=16)
    plt.ylabel( "reward", fontsize=16)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
