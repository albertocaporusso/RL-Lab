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

def training_loop( env, actor_net, critic_net, updateRule, frequency, episodes=100 ):
    actor_opt = tf.keras.optimizers.Adam()
    critic_opt = tf.keras.optimizers.Adam()

    rewards_list, reward_queue = [], collections.deque( maxlen=100 )
    memory_buffer = []
    for ep in range(episodes):
        state = env.reset()[0]
        state = state.reshape(-1,6)
        ep_reward = 0
        ep_lenght = 0
        while True:

            #TODO: select the action to perform
            distribution = actor_net(state).numpy()[0]
            action = np.random.choice(3, p = distribution)

            #TODO: Perform the action, store the data in the memory buffer and update the reward
            next_state, reward, terminated,truncated,_ = env.step(action)
            done = terminated or truncated
            next_state = np.asarray(next_state).reshape(-1,6)



            memory_buffer.append([state,action,next_state,reward,done])
            ep_reward += reward
            ep_lenght +=1

            #TODO: exit condition for the episode
            if done: break

            #TODO: update the current state
            state = next_state

        #TODO: Perform the actual training every 'frequency' episodes


        if ep % frequency == 0 and ep!=0:
            updateRule( actor_net, critic_net, memory_buffer, actor_opt, critic_opt )
            memory_buffer = []


        # Update the reward list to return
        reward_queue.append( ep_reward )
        rewards_list.append( np.mean(reward_queue) )
        print( f"episode {ep:4d}: rw: {int(ep_reward):3d} (averaged: {np.mean(reward_queue):5.2f}) num_steps: {ep_lenght} " )

    # Close the enviornment and return the rewards list
    env.close()
    return rewards_list



def A2C( actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer, gamma=0.99 ):

    """
    Main update rule for the A2C update. This function includes the updates for the actor network (or policy function)
    and for the critic network (or value function)

    """
    #TODO: implement the update rule for the critic (value function)
    memory_buffer = np.asarray(memory_buffer)
    for _ in range(10):
        # Shuffle the memory buffer
        np.random.shuffle(memory_buffer)
        #TODO: extract the information from the buffer

        state = np.vstack(memory_buffer[:,0])
        action = np.array(memory_buffer[:,1], dtype = int)
        next_state = np.vstack(memory_buffer[:,2])
        reward = memory_buffer[:,3]
        done = memory_buffer[:,4]

        with tf.GradientTape() as critic_tape:

            #TODO: Compute the target and the MSE between the current prediction
            ## and the expected advantage

            #target = reward + (1-int(done))*gamma*critic_net(next_state)
            target = reward + (1-done.astype(int))*gamma*critic_net(next_state).numpy()
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
        target = tf.math.log(probs)*(adv_a - adv_b)

        #TODO: compute the final objective to optimize, is the average between all the considered trajectories

        objective = -tf.math.reduce_mean(target)
        grad_act = actor_tape.gradient(objective, actor_net.trainable_variables)
        actor_optimizer.apply_gradients(zip(grad_act, actor_net.trainable_variables))



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



# def training_loop( env, neural_net, updateRule, frequency, episodes=100 ):
#     """
#     Main loop of the reinforcement learning algorithm. Execute the actions and interact
#     with the environment to collect the experience for the training.

#     Args:
#         env: gymnasium environment for the training
#         neural_net: the model to train
#         updateRule: external function for the training of the neural network

#     Returns:
#         averaged_rewards: array with the averaged rewards obtained

#     """

#     #TODO: initialize the optimizer
#     optimizer = tf.keras.optimizers.Adam()
#     rewards_list, reward_queue = [], collections.deque( maxlen=100 )
#     memory_buffer = []
#     for ep in range(episodes):
#         state = env.reset()[0]
#         state = np.asarray(state)
#         state = state.reshape(-1,6)
#         ep_reward = 0
#         episode = []
#         n_steps = 0
#         while True:

#             #TODO: select the action to perform
#             distribution = neural_net(state).numpy()[0]
#             action = np.random.choice(3, p = distribution)

#             #TODO: Perform the action, store the data in the memory buffer and update the reward
#             if ep == 4990:
#                 next_state, reward, terminated,truncated,_ = env.step(action)
#             else:
#                 next_state, reward, terminated,truncated,_ = env.step(action)
#             done = terminated or truncated
#             next_state = next_state.reshape(-1,6)

#             episode.append([state, action, next_state, reward, done])
#             ep_reward += reward
#             n_steps+=1

#             #TODO: exit condition for the episode
#             if done: break

#             #TODO: update the current state
#             state = next_state
#         memory_buffer.append(np.asarray(episode))
#         #TODO: Perform the actual training every 'frequency' episodes
#         if ep % frequency == 0:
#             updateRule( neural_net,memory_buffer, optimizer )
#             memory_buffer = []


#         # Update the reward list to return
#         reward_queue.append( ep_reward )
#         rewards_list.append( np.mean(reward_queue) )
#         print( f"episode {ep:4d}: rw: {int(ep_reward):3d} (averaged: {np.mean(reward_queue):5.2f})  len_ep : {n_steps}" )

#     # Close the enviornment and return the rewards list
#     env.close()
#     return rewards_list

# def REINFORCE_rw2go( neural_net, memory_buffer, optimizer ):
#     """
#     Main update rule for the REINFORCE process, with the addition of the reward-to-go trick,

#     """
#     memory_buffer = np.asarray(memory_buffer)
#     r = []
#     for ep in memory_buffer:
#         r.append(np.flip(np.cumsum(ep[:,3])))
#     with tf.GradientTape() as tape:

#         objectives= []
#         for i,ep in enumerate(memory_buffer):
#             state = np.vstack(ep[:,0])
#             action = np.vstack(ep[:,1])
#             action = action[:,0]
#             probabilities = neural_net(state)
#             probability = []
#             for ind, a in enumerate(action):
#                 probability.append(probabilities[ind][a])
#             target = tf.math.reduce_sum(tf.math.log(probability)*r[i])
#             objectives.append(target)

#         objective = -tf.math.reduce_mean(objectives)
#         grads = tape.gradient(objective, neural_net.trainable_variables)
#         optimizer.apply_gradients(zip(grads, neural_net.trainable_variables))

def main():
    print( "\n***************************************************" )
    print( "*  Welcome to the eleventh lesson of the RL-Lab!  *" )
    print( "*                 (DRL in Practice)               *" )
    print( "***************************************************\n" )

    _training_steps = 2500

    # Crete the environment and add the wrapper for the custom reward function
    # gymnasium.envs.register(
    # 	id='MountainCarMyVersion-v0',
    # 	entry_point='gymnasium.envs.classic_control:MountainCarEnv',
    # 	max_episode_steps=1000
    # )
    env = gymnasium.make("Acrobot-v1")
    env = OverrideReward(env)

    # Create the networks and perform the actual training
    actor_net = createDNN( 6, 3, nLayer=4, nNodes=32, last_activation = "softmax")
    critic_net = createDNN( 6, 1, nLayer=4, nNodes=32, last_activation = "linear")
    a2c = training_loop( env, actor_net,critic_net, A2C, frequency=1, episodes=_training_steps  )

    # Save the trained neural network
    #actor_net.save( "MountainCarActor.h5" )

    # Plot the results
    t = np.arange(0, _training_steps)
    plt.plot(t, a2c, label="A2C", linewidth=3)
    plt.xlabel( "episodes", fontsize=16)
    plt.ylabel( "rewards", fontsize=16)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()