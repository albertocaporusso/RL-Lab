import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# TODO: implement the following functions as in the previous lessons
# Notice that the value function has only one output with a linear activation
# function in the last layer
class OverrideReward( gymnasium.wrappers.NormalizeReward ):
    """
    Gymansium wrapper useful to update the reward function of the environment

    """
    def step(self, action):
            #previous_observation = np.array(self.env.state, dtype=np.float32)
            observation, reward, terminated, truncated, info = self.env.step( action )

            if observation in [5,7,11,12]:
                reward = -1000
            elif observation == 15:
                reward = 1000
            else:
                current_row, current_col = np.unravel_index(observation, (self.env.nrow, self.env.ncol))
                distance = abs(current_row - 3) + abs(current_col - 3)
                reward = 6 - distance

            return observation, reward, terminated, truncated, info

def createDNN( nInputs, nOutputs, nLayer, nNodes, last_activation ):
    model = Sequential()

    model.add(Dense(nNodes,input_dim = nInputs, activation = 'relu'))
    for _ in range(nLayer - 1):
        model.add(Dense(nNodes, activation = "relu"))
    model.add(Dense(nOutputs, activation = last_activation))

    return model

def training_loop( env, actor_net, critic_net, updateRule, frequency=10, episodes=100 ):
    actor_opt = tf.keras.optimizers.Adam()
    critic_opt = tf.keras.optimizers.Adam()
    dim_state = 1
    n_action = env.action_space.n
    rewards_list, reward_queue = [], collections.deque( maxlen=100 )
    memory_buffer = []
    term=0
    for ep in range(episodes):
        state = env.reset()[0]
        state = np.array(state).reshape(-1)
        ep_reward = 0
        ep_len = 0
        while True:

            #TODO: select the action to perform
            distribution = actor_net(state).numpy()[0]
            action = np.random.choice(n_action, p = distribution)

            #TODO: Perform the action, store the data in the memory buffer and update the reward
            next_state, reward, terminated,truncated,_ = env.step(action)
            done = terminated or truncated
            next_state = np.array(next_state).reshape(-1)
            if next_state == 15:
                term=term + 1


            memory_buffer.append([state,action,next_state,reward,done])

            ep_reward += reward
            ep_len +=1

            #TODO: exit condition for the episode
            if done:
                break

            #TODO: update the current state
            state = next_state

        #TODO: Perform the actual training every 'frequency' episodes


        if ep % frequency == 0 and ep!=0:
            updateRule( actor_net, critic_net, memory_buffer, actor_opt, critic_opt )
            memory_buffer = []


        # Update the reward list to return
        reward_queue.append( ep_reward )
        rewards_list.append( np.mean(reward_queue) )
        print( f"episode {ep:4d}: rw: {(ep_reward):3f} (averaged: {np.mean(reward_queue):5.2f})   ep_len: {ep_len}   terminated : {term}" )

    # Close the enviornment and return the rewards list
    env.close()
    return rewards_list



def A2C( actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer, gamma=0.99 ):

    """
    Main update rule for the A2C update. This function includes the updates for the actor network (or policy function)
    and for the critic network (or value function)

    """
    memory_buffer = np.asarray(memory_buffer)
    #TODO: implement the update rule for the critic (value function)
    for _ in range(10):
        # Shuffle the memory buffer
        np.random.shuffle(memory_buffer)
        #TODO: extract the information from the buffer



        state = np.vstack(memory_buffer[:,0])
        action = np.array(memory_buffer[:,1], dtype = int)
        next_state = np.vstack(memory_buffer[:,2])
        reward = memory_buffer[:,3]
        done = memory_buffer[:,4]
        # Tape for the critic


        with tf.GradientTape() as critic_tape:

            #TODO: Compute the target and the MSE between the current prediction
            ## and the expected advantage
            target = reward + (1-done.astype(int))*gamma*critic_net(next_state)
            #target = reward + (1-int(done))*gamma*critic_net(next_state)
            pred = critic_net(state)
            mse = tf.math.square(pred - target)
            #mse = tf.math.reduce_mean(mse)

            #TODO: Perform the actual gradient-descent process
            grad_crit = critic_tape.gradient(mse, critic_net.trainable_variables)
            critic_optimizer.apply_gradients(zip(grad_crit, critic_net.trainable_variables))


    #TODO: implement the update rule for the actor (policy function)
    #TODO: extract the information from the buffer for the policy update
    # Tape for the actor
    with tf.GradientTape() as actor_tape:
        #TODO: compute the log-prob of the current trajectory and
        # the objective function, notice that:
        # the REINFORCE objective is the sum of the logprob (i.e., the probability of the trajectory)
        # multiplied by advantage
        #state, action, next_state, reward, done = memory_buffer
        probabilities = actor_net(state)
        indices = tf.transpose(tf.stack([tf.range(probabilities.shape[0]), action]))
        probs = tf.gather_nd(
            indices=indices,
            params=probabilities
        )
        adv_b = critic_net(state).numpy().reshape(-1)

        adv_a = reward + gamma*critic_net(next_state).numpy().reshape(-1)
        target = (tf.math.log(probs))*(adv_a - adv_b)

        #TODO: compute the final objective to optimize, is the average between all the considered trajectories
        objective = - tf.math.reduce_sum(target)

        grad_act = actor_tape.gradient(objective, actor_net.trainable_variables)
        actor_optimizer.apply_gradients(zip(grad_act, actor_net.trainable_variables))


def main():
    print( "\n*************************************************" )
    print( "*  Welcome to the tenth lesson of the RL-Lab!   *" )
    print( "*                    (A2C)                      *" )
    print( "*************************************************\n" )

    _training_steps = 2000

    env = gymnasium.make( "FrozenLake-v1",desc=None, map_name="4x4", is_slippery=False )
    env = OverrideReward(env)
    dim_state =1
    n_action = env.action_space.n
    actor_net = createDNN( dim_state, n_action, nLayer=3, nNodes=32, last_activation="softmax")
    critic_net = createDNN( dim_state, 1, nLayer=3, nNodes=32, last_activation="linear")
    reward_A2C = training_loop( env, actor_net, critic_net, A2C, frequency=5, episodes=_training_steps  )

    t = np.arange(0, _training_steps)
    plt.plot(t, reward_A2C, label="A2C", linewidth=3)
    plt.xlabel( "epsiodes", fontsize=16)
    plt.ylabel( "reward", fontsize=16)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
