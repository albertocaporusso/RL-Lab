import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections


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

    rewards_list, reward_queue = [], collections.deque( maxlen=100 )
    memory_buffer = []
    for ep in range(episodes):
        state = env.reset()[0]
        state = state.reshape(-1,2)
        ep_reward = 0
        ep_lenght = 0
        while True:

            distribution = actor_net(state).numpy()[0]
            action = np.random.choice(3, p = distribution)

            #TODO: Perform the action, store the data in the memory buffer and update the reward
            next_state, reward, terminated,truncated,_ = env.step(action)
            done = terminated or truncated
            next_state = next_state.reshape(-1,2)

            memory_buffer.append([state,action,next_state,reward,done])
            ep_reward += reward
            ep_lenght +=1


            if done: break

            state = next_state

        if ep % frequency == 0:
            updateRule( actor_net, critic_net, memory_buffer, actor_opt, critic_opt )
            memory_buffer = []


        # Update the reward list to return
        reward_queue.append( ep_reward )
        rewards_list.append( np.mean(reward_queue) )
        print( f"episode {ep:4d}: rw: {int(ep_reward):3d} (averaged: {np.mean(reward_queue):5.2f}) episode lenght: {ep_lenght}" )

    # Close the enviornment and return the rewards list
    env.close()
    return rewards_list



def A2C( actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer, gamma=0.99 ):

    """
    Main update rule for the A2C update. This function includes the updates for the actor network (or policy function)
    and for the critic network (or value function)

    """
    #implement the update rule for the critic (value function)
    memory_buffer = np.asarray(memory_buffer)
    for _ in range(10):
        # Shuffle the memory buffer
        np.random.shuffle(memory_buffer)
        #extract the information from the buffer



        state = np.vstack(memory_buffer[:,0])
        action = np.array(memory_buffer[:,1], dtype = int)
        next_state = np.vstack(memory_buffer[:,2])
        reward = memory_buffer[:,3]
        done = memory_buffer[:,4]

        # Tape for the critic
        with tf.GradientTape() as critic_tape:

            #Compute the target and the MSE between the current prediction
            ## and the expected advantage
            target = reward + (1-done.astype(int))*gamma*critic_net(next_state)

            pred = critic_net(state)
            mse = tf.math.square(pred - target)
            #mse = tf.math.reduce_mean(mse)

            #Perform the actual gradient-descent process
            grad_crit = critic_tape.gradient(mse, critic_net.trainable_variables)
            critic_optimizer.apply_gradients(zip(grad_crit, critic_net.trainable_variables))


    #implement the update rule for the actor (policy function)
    #extract the information from the buffer for the policy update
    # Tape for the actor

    with tf.GradientTape() as actor_tape:
        #compute the log-prob of the current trajectory and
        # the objective function, notice that:
        # the REINFORCE objective is the sum of the logprob (i.e., the probability of the trajectory)
        # multiplied by advantage
        probabilities = actor_net(state)
        indices = tf.transpose(tf.stack([tf.range(probabilities.shape[0]), action]))
        probs = tf.gather_nd(
            indices=indices,
            params=probabilities
        )
        adv_b = critic_net(state).numpy().reshape(-1)

        adv_a = reward + gamma*critic_net(next_state).numpy().reshape(-1)
        #adv_a = reward + (1-done.astype(int))*gamma*critic_net(next_state).numpy().reshape(-1)
        target = tf.math.log(probs)*(adv_a - adv_b)

        #compute the final objective to optimize, is the average between all the considered trajectories
        objective = -tf.math.reduce_sum(target)

        grad_act = actor_tape.gradient(objective, actor_net.trainable_variables)
        actor_optimizer.apply_gradients(zip(grad_act, actor_net.trainable_variables))

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
    actor_net = createDNN( 2, 3, nLayer=2, nNodes=32, last_activation="softmax")

    critic_net = createDNN( 2, 1, nLayer=2, nNodes=32, last_activation="linear")
    rewards_training = training_loop( env, actor_net, critic_net, A2C, frequency=1, episodes=_training_steps  )

    # Save the trained neural network
    actor_net.save( "MountainCarActor.h5" )

    # Plot the results
    t = np.arange(0, _training_steps)
    plt.plot(t, rewards_training, label="A2C", linewidth=3)
    plt.xlabel( "epsiodes", fontsize=16)
    plt.ylabel( "length", fontsize=16)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()