import os, sys, numpy, random
module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path: sys.path.append(module_path)
from DangerousGridWorld import GridWorld


def epsilon_greedy(q, state, epsilon):
	"""
	Epsilon-greedy action selection function

	Args:
		q: q table
		state: agent's current state
		epsilon: epsilon parameter

	Returns:
		action id
	"""
	if numpy.random.random() < epsilon:
		return numpy.random.choice(q.shape[1])
	return q[state].argmax()


def dynaQ( environment, maxiters=250, n=10, eps=0.3, alfa=0.3, gamma=0.99 ):

	Q = numpy.zeros((environment.observation_space, environment.action_space))
	M = numpy.array([[[None, None] for _ in range(environment.action_space)] for _ in range(environment.observation_space)])
	memory =  []
	for _ in range(maxiters) :
		s = environment.random_initial_state()
		a = epsilon_greedy(Q, s, eps)
		s_1 = environment.sample(a,s)
		memory.append(list([s,a])) #salvo a coppie "s" e "a" che vengono visitati
		r = environment.R[s_1]
		Q[s][a] = Q[s][a] + alfa*(r+gamma*max(Q[s_1]) - Q[s][a])
		M[s][a] = (r, s_1)
		for _ in range(n) :
			#select randomly previous visited pair state-action
			index = random.choice(range(len(memory)))
			s,a = memory[index]
			r, s_1 = M[s][a]
			Q[s][a] = Q[s][a] + alfa*(r+gamma*max(Q[s_1]) - Q[s][a])

	policy = Q.argmax(axis=1)
	return policy



def main():
	print( "\n************************************************" )
	print( "*   Welcome to the fifth lesson of the RL-Lab!   *" )
	print( "*                  (Dyna-Q)                      *" )
	print( "**************************************************" )

	print("\nEnvironment Render:")
	env = GridWorld( deterministic=True )
	env.render()

	print( "\n6) Dyna-Q" )
	dq_policy_n00 = dynaQ( env, n=0  )
	dq_policy_n25 = dynaQ( env, n=25 )
	dq_policy_n50 = dynaQ( env, n=50 )

	env.render_policy( dq_policy_n50 )
	print()
	print( f"\tExpected reward with n=0 :", env.evaluate_policy(dq_policy_n00) )
	print( f"\tExpected reward with n=25:", env.evaluate_policy(dq_policy_n25) )
	print( f"\tExpected reward with n=50:", env.evaluate_policy(dq_policy_n50) )



if __name__ == "__main__":
	main()