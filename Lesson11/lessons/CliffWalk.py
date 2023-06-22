import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# TODO: implement the following functions as in the previous lessons
def createDNN( nInputs, nOutputs, nLayer, nNodes):
    model = Sequential()

    model.add(Dense(nNodes,input_dim = nInputs, activation = 'relu'))
    for _ in range(nLayer - 1):
        model.add(Dense(nNodes, activation = "relu"))
    model.add(Dense(nOutputs, activation = "linear"))

    return model

def mse( network, dataset_input, target ):
	"""
	Compute the MSE loss function

	"""

	# Compute the predicted value, over time this value should
	# looks more like to the expected output (i.e., target)
	predicted_value = network( dataset_input )

	# Compute MSE between the predicted value and the expected labels
	mse = tf.math.square(predicted_value - target)
	mse = tf.math.reduce_mean(mse)

	# Return the averaged values for computational optimization
	return mse

def training_loop( env, neural_net, updateRule, eps=1, episodes=100, updates=1 ):
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

    optimizer = tf.keras.optimizers.Adam()
    rewards_list, memory_buffer = [], collections.deque( maxlen=1000 )
    averaged_rewards = []
    success = 0
    for ep in range(episodes):
        state = env.reset()[0]
        state = np.asarray(state).reshape(-1)
        ep_reward = 0
        ep_length = 0
        while True:

            ep_length+=1
            if np.random.random() > eps:
                action = env.action_space.sample()
            else :
                action = neural_net(state).numpy().argmax()

            next_state, reward, terminated,truncated,_ = env.step(action)
            if next_state == 47:
                success += 1
            next_state = np.asarray(next_state).reshape(-1)
            done = terminated or truncated
            memory_buffer.append([state, action,next_state, reward,  done])
            ep_reward += reward

            # Perform the actual training

            updateRule( neural_net, memory_buffer, optimizer )


            #TODO: exit condition for the episode

            if done:
                break


            #TODO: update the current state
            state = next_state
        # Update the reward list to return
        rewards_list.append( ep_reward )
        averaged_rewards.append( np.mean(rewards_list) )
        print( f"episode {ep:2d}: rw: {averaged_rewards[-1]:3.2f}, eps: {eps:3.2f},  ep_length: {ep_length}, success_eps: {success}, reward_laststate: {next_state, reward}" )
        eps = eps*0.99

    # Close the enviornment and return the rewards list
    env.close()
    return averaged_rewards

def DQNUpdate( neural_net, memory_buffer, optimizer, batch_size=32, gamma=0.99 ):

    """
    Main update rule for the DQN process. Extract data from the memory buffer and update
    the newtwork computing the gradient.

    """

    if len(memory_buffer) < batch_size: return

    indices = np.random.randint( len(memory_buffer), size=batch_size)
    for idx in indices:

        #TODO: extract data from the buffer

        state ,action,next_state,reward,done = memory_buffer[idx]

        #TODO: compute the target for the training
        target = neural_net(state).numpy()
        if done:
            target[0][action] = reward
        else:
            maxq = max(neural_net(next_state).numpy()[0])
            target[0][action] = reward + (maxq*gamma)

        #TODO: compute the gradient and perform the backpropagation step
        with tf.GradientTape() as tape:
            objective = mse( neural_net, state, target )
            grad = tape.gradient(objective, neural_net.trainable_variables)
            optimizer.apply_gradients(zip(grad, neural_net.trainable_variables))


# TODO: implement the following class
class OverrideReward( gymnasium.wrappers.NormalizeReward ):
    """
    Gymansium wrapper useful to update the reward function of the environment

    """

    def step(self, action):

            observation, reward, terminated, truncated, info = self.env.step( action )
            if observation == 47:
                reward = 200
            elif observation in [37,38,39,40,41,42,43,44,45,46]:
                reward = -100
            else:
                reward = -1
            return observation, reward, terminated, truncated, info




def main():
    print( "\n***************************************************" )
    print( "*  Welcome to the eleventh lesson of the RL-Lab!  *" )
    print( "*                 (DRL in Practice)               *" )
    print( "***************************************************\n" )

    _training_steps = 2500


    env = gymnasium.make("CliffWalking-v0")
    env = OverrideReward(env)

    # Create the networks and perform the actual training
    neural_net = createDNN( 1, 4, nLayer=2, nNodes=32)
    dqn = training_loop( env, neural_net, DQNUpdate, episodes=_training_steps  )


    # Plot the results
    t = np.arange(0, _training_steps)
    plt.plot(t, dqn, label="DQN", linewidth=3)
    plt.xlabel( "episodes", fontsize=16)
    plt.ylabel( "rewards", fontsize=16)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()