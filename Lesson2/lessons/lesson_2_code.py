import os, sys
module_path = os.path.abspath(os.path.join('../tools'))
if module_path not in sys.path: sys.path.append(module_path)
from DangerousGridWorld import GridWorld


def value_iteration(environment, maxiters=300, discount=0.9, max_error=1e-3):
    """
    Performs the value iteration algorithm for a specific environment

    Args:
        environment: OpenAI Gym environment
        maxiters: timeout for the iterations
        discount: gamma value, the discount factor for the Bellman equation
        max_error: the maximum error allowd in the utility of any state

    Returns:
        policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
    """

    U_1 = [0 for _ in range(environment.observation_space)]
    delta = 0
    U = U_1.copy()
    i = 0
    while True:
        i+=1
        U = U_1.copy()
        delta = 0

        for state in range(environment.observation_space):

            val = [0 for _ in environment.actions]
            for action in environment.actions:
                for next_state in range(environment.observation_space):
                    val[action] += environment.transition_prob(state,action,next_state) * U[next_state]

            if environment.is_terminal(state):
                U_1[state] = environment.R[state]
            else : U_1[state] = environment.R[state] + (discount * max(val))

            delta = max(abs(U_1[state] - U[state]),delta)

        if ( delta < ((max_error * (1 - discount))/discount) ) or (i > 300): break
    return environment.values_to_policy( U )



def policy_iteration(environment, maxiters=300, discount=0.9, maxviter=10):
    """
    Performs the policy iteration algorithm for a specific environment

    Args:
        environment: OpenAI Gym environment
        maxiters: timeout for the iterations
        discount: gamma value, the discount factor for the Bellman equation
        maxviter: number of epsiodes for the policy evaluation

    Returns:
        policy: 1-d dimensional array of action identifiers where index `i` corresponds to state id `i`
    """

    policy = [0 for _ in range(environment.observation_space)] #initial policy
    U = [0 for _ in range(environment.observation_space)] #utility array
    i = 0

    while True:
        i+=1
        # Step (1): Policy Evaluation
        for _ in range(maxviter): # capisco perchè cambia ma la policy come posso cambiarla ? rimane sempre la stessa ! in algo policy è anche lei in i
            U_1 = U
            for state in range(environment.observation_space):
                pval = 0
                for next_state in range(environment.observation_space):
                    # Come cazzo faccio a usare "i" policy se ne tengo solo una e viene calcolata nello step succesivo
                    pval += environment.transition_prob(state,policy[state],next_state) * U_1[next_state]

                if environment.is_terminal(state):
                    U[state] = environment.R[state]
                else:
                    U[state] = environment.R[state] + (discount * pval)

        # Step (2) Policy Improvement
        unchanged = True
        for state in range(environment.observation_space):

            # FIRST ARGUMENT
            sumOfProbabilites= [0 for _ in environment.actions]
            for action in environment.actions:
                val = 0
                for next_state in range(environment.observation_space):
                    val += environment.transition_prob(state,action,next_state) * U[next_state]
                sumOfProbabilites[action]= (val,action)

            # SECOND ARGUMENT
            val2 = 0
            for next_state in range(environment.observation_space):
                val2 += environment.transition_prob(state,policy[state],next_state) * U[next_state]

            if max(sumOfProbabilites)[0] > val2:
                policy[state] =  max(sumOfProbabilites)[1]
                unchanged= False

        if unchanged or i>maxiters: break

    return policy




def main():
    print( "\n************************************************" )
    print( "*  Welcome to the second lesson of the RL-Lab! *" )
    print( "*    (Policy Iteration and Value Iteration)    *" )
    print( "************************************************" )

    print("\nEnvironment Render:")
    env = GridWorld()

    print( "\n1) Value Iteration:" )
    vi_policy = value_iteration( env )
    env.render_policy( vi_policy )
    print( "\tExpected reward following this policy:", env.evaluate_policy(vi_policy) )

    print( "\n1) Policy Iteration:" )
    pi_policy = policy_iteration( env )
    env.render_policy( pi_policy )
    print( "\tExpected reward following this policy:", env.evaluate_policy(pi_policy) )


if __name__ == "__main__":
    main()