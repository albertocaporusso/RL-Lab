import os, sys, numpy, random
module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path: sys.path.append(module_path)
from DangerousGridWorld import GridWorld


def random_dangerous_grid_world( environment ):
	"""
	Performs a random trajectory on the given Dangerous Grid World environment

	Args:
		environment: OpenAI Gym environment

	Returns:
		trajectory: an array containing the sequence of states visited by the agent
	"""
	trajectory = []
	state = environment.start_state

	for _ in range(10):
		rnd_move = random.randint(0,3)
		state = environment.sample(rnd_move, state)
		trajectory.append(state)
		if (environment.is_terminal(state)) :
			break
	return trajectory


class RecyclingRobot():
	"""
	Class that implements the environment Recycling Robot of the book: 'Reinforcement
	Learning: an introduction, Sutton & Barto'. Example 3.3 page 52 (second edition).

	Attributes
	----------
		observation_space : int
			define the number of possible actions of the environment
		action_space: int
			define the number of possible states of the environment
		actions: dict
			a dictionary that translate the 'action code' in human languages
		states: dict
			a dictionary that translate the 'state code' in human languages

	Methods
	-------
		reset( self )
			method that reset the environment to an initial state; returns the state
		step( self, action )
			method that perform the action given in input, computes the next state and the reward; returns
			next_state and reward
		render( self )
			method that print the internal state of the environment
	"""


	def __init__( self ):

		# Loading the default parameters
		self.alfa = 0.7
		self.beta = 0.7
		self.r_search = 0.5
		self.r_wait = 0.2
		self.r_rescue = -3

		# Defining the environment variables
		self.observation_space = 2
		self.action_space = 3
		self.actions = {0: 'SEARCH', 1: 'WAIT', 2: 'RECHARGE'}
		self.states = {0 :'HIGH',1: 'LOW'}


	def reset( self ):
		self.state = 0 #ricarica robot
		return self.state


	def step( self, action ):

		reward = 0
		if action == 1 : #wait
			reward = self.r_wait
		elif action == 0 : #search
			if self.state == 0 :

				next_state = random.choices(self.states, weights=(self.alfa, (1-self.alfa)), k=1)
				#print("Next state: ", next_state)
				if next_state[0] == 0:
					self.state = 0
				else:
					self.state = 1
				reward = self.r_search
			else :
				next_state = random.choices(self.states, weights=((1-self.beta), self.beta), k=1)
				#print("Rescue: ", next_state)
				if next_state[0] == 0 :
					reward = self.r_rescue
					self.state = 0
				else :
					reward = self.r_search
					self.state = 1
		elif action == 2 : #recharge
			self.state=0
		else:
			print("Error")





		return self.state, reward, False, None


	def render( self ):
		for i in range(self.observation_space):
			if i == 1 : print("[H]", end ="")
			elif i==2 : print("[L]", end = "")
		return True


def main():
	print( "\n************************************************" )
	print( "*  Welcome to the first lesson of the RL-Lab!  *" )
	print( "*             (MDP and Environments)           *" )
	print( "************************************************" )

	print( "\nA) Random Policy on Dangerous Grid World:" )
	env = GridWorld()
	env.render()
	random_trajectory = random_dangerous_grid_world( env )
	print( "\nRandom trajectory generated:", random_trajectory )


	print( "\nB) Custom Environment: Recycling Robot" )
	env = RecyclingRobot()
	state = env.reset()
	ep_reward = 0
	for step in range(10):
		a = numpy.random.randint( 0, env.action_space )
		new_state, r, _, _ = env.step( a )
		ep_reward += r
		print( f"\tFrom state '{env.states[state]}' selected action '{env.actions[a]}': \t total reward: {ep_reward:1.1f}" )
		state = new_state


if __name__ == "__main__":
	main()