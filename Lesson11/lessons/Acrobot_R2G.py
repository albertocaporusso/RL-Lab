import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# TODO: implement the following functions as in the previous lessons
def createDNN( nInputs, nOutputs, nLayer, nNodes, last_activation ):
    model = Sequential()

    model.add(Dense(nNodes,input_dim = nInputs, activation = 'relu'))
    for _ in range(nLayer - 1):
        model.add(Dense(nNodes, activation = "relu"))
    model.add(Dense(nOutputs, activation = last_activation))

    return model

# TODO: implement the following class
class OverrideReward( gymnasium.wrappers.NormalizeReward ):
    """
    Gymansium wrapper useful to update the reward function of the environment

    """

    def step(self, action):

            observation, reward, terminated, truncated, info = self.env.step( action )
            #we modify the reward based on some custom logic. In this example,
            # we add 1.0 to the reward if the first link of the pendulum is above 0.5 and the second link is above 0.5,
            # and subtract 1.0 from the reward if the first link is below -0.5 and the second link is below -0.5.

            if observation[0] > 0.5 and observation[1] > 0.5:
                reward += 1.0
            elif observation[0] < -0.5 and observation[1] < -0.5:
                reward -= 1.0
            return observation, reward, terminated, truncated, info



def training_loop( env, neural_net, updateRule, frequency, episodes=100 ):
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
        state = np.asarray(state)
        state = state.reshape(-1,6)
        ep_reward = 0
        episode = []
        n_steps = 0
        while True:

            #TODO: select the action to perform
            distribution = neural_net(state).numpy()[0]
            action = np.random.choice(3, p = distribution)

            #TODO: Perform the action, store the data in the memory buffer and update the reward
            if ep == 4990:
                next_state, reward, terminated,truncated,_ = env.step(action)
            else:
                next_state, reward, terminated,truncated,_ = env.step(action)
            done = terminated or truncated
            next_state = next_state.reshape(-1,6)

            episode.append([state, action, next_state, reward, done])
            ep_reward += reward
            n_steps+=1

            #TODO: exit condition for the episode
            if done: break

            #TODO: update the current state
            state = next_state
        memory_buffer.append(np.asarray(episode))
        if ep == 150:
            env = gymnasium.make("Acrobot-v1", render_mode = 'human')
            env = OverrideReward(env)
        #TODO: Perform the actual training every 'frequency' episodes
        if ep % frequency == 0:
            updateRule( neural_net,memory_buffer, optimizer )
            memory_buffer = []


        # Update the reward list to return
        reward_queue.append( ep_reward )
        rewards_list.append( np.mean(reward_queue) )
        print( f"episode {ep:4d}: rw: {int(ep_reward):3d} (averaged: {np.mean(reward_queue):5.2f})  len_ep : {n_steps}" )

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

    _training_steps = 2500

    env = gymnasium.make("Acrobot-v1")
    env = OverrideReward(env)
    dim_state = env.observation_space.shape[0]
    n_action = env.action_space.n
    # Create the networks and perform the actual training
    net = createDNN( dim_state, n_action, nLayer=4, nNodes=32, last_activation = "softmax")

    r2g = training_loop( env, net, REINFORCE_rw2go, frequency=1, episodes=_training_steps  )

    # Save the trained neural network
    #actor_net.save( "MountainCarActor.h5" )

    # Plot the results
    t = np.arange(0, _training_steps)
    plt.plot(t, r2g, label="R2G", linewidth=3)
    plt.xlabel( "episodes", fontsize=16)
    plt.ylabel( "rewards", fontsize=16)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()