import numpy as np
import math as math
import matplotlib.pyplot as plt

def value_function_analytical(A,B, Kapital_Grid):
    v = np.array([0.]*Kapital_Grid.size)
    for i in list(range(0,Kapital_Grid.size)):
        v[i] = v_analytical(Kapital_Grid[i],A,B)
    return(v)

def policy_function_analytical(alpha,beta,A,Kapital_Grid):
    v = np.array([0.]*Kapital_Grid.size)
    for i in list(range(0,Kapital_Grid.size)):
        v[i] = (Kapital_Grid[i] ** alpha) * beta * alpha * A
    return(v)

def policy_function_from_policy_index(policy_index,Kapital_Grid):
    policy_function = np.array([0.]*Kapital_Grid.size)
    for i in list(range(0,policy_function.size)):
        policy_function[i] = Kapital_Grid[int(policy_index[i])]
    return(policy_function)
        

def value_function_iteration(alpha, beta, technology, Kapital_Grid,tolerance):
    difference = 2 * tolerance
    lenght_of_grid = Kapital_Grid.size
    v = np.array([0.]*lenght_of_grid)              # vector in which we safe the current itteration
    v_prime = np.array([0.]*lenght_of_grid)        # vector in which we safe the previous itteration
    Dummy = np.array([0.]*lenght_of_grid)
    policy_index = np.array([0.]*lenght_of_grid)
    while difference > tolerance:                                                             # we iterate as long we are above the tolerance epsilon
        for k in list(range(0,Kapital_Grid.size)):                                          # for each possible kapital value 'Capital(k)' we compute the v_n+1(k) (which is v_prime here)
            for j in list(range(0,Kapital_Grid.size)):                                      # in order to do that we need to find the maximum of the term in the next line
                if(Kapital_Grid[j] < f(Kapital_Grid[k])):                                   # for those 'capital(k)' that are smaller than f(capital(k)) otherwise the log is not defined
                    Dummy[j] = math.log(technology(Kapital_Grid[k])-Kapital_Grid[j]) + beta * v[j]
                else: Dummy[j] = None
            
            v_prime[k] = np.max(Dummy)#These lines are just technical stuff. could probably be done more compactly.
            policy_index[k] = np.argmax(Dummy)
        difference = np.max(np.absolute(v-v_prime))     # We update the difference to check later if we need to stop the loop
    
        for k in list(range(0,v.size)):         # Here we set the values of v equal to the values of v_prime
            v[k] = v_prime[k]                   # v = v_prime does not work because v and v_prime are pointers (I think)
        #l= l +1                                # we do that because we need 'v_n' = v to compute 'v_n+1' = v_prime in the next iteration
    return([v,policy_index])                          # the policy function that is returned here does not give you for capital 'k' capital 'k_prime'
                                                # instead it takes as an input the index of 'k' (in Kapital_Grid) and as output it gives you the index of 'k_prime' 
    
def find_optimal_path(k_0,n,policy_index,Kapital_Grid):   # k_0 must be the index of the start capital!
    path_index = np.array([0.]*n)
    path_index[0] = k_0
    
    for i in list(range(1,n)):                            # We first write the indexes of the optimal path into path_index
        path_index[i]=policy_index[int(path_index[i-1])]
    for i in list(range(1,n)):                            # then we 'translate' the indexes into the corresponding values
        path_index[i] = Kapital_Grid[int(path_index[i])]       
    return(path_index)





def v_analytical(k,A,B):
    return (A + B * math.log(k))



Kapital_Grid = np.arange(2,10,0.1)         # the discrete grid for capital values
epsilon = 10**(-6)           # variables as stated in the exercise
alpha = 0.3
beta = 0.6
A = 20
n = 20                      # number of steps of optimal kapital path to be
k_0 = 0                     # The index of the initial kapital in the Kapital_Grid, for which optimal path should be determined
alpha_0 = (math.log(A * (1 - alpha * beta))) / (1 - beta) + (beta * alpha * math.log(alpha * beta * A)) / ((1 - beta)*(1 - alpha * beta))
alpha_1 = (alpha) / (1 - alpha * beta)



def f(x):                    # the production function as stated in the exercise
    return A*(x ** alpha)
    
def u(c):                    # the ustility function as stated in the exercise
    return math.log(c)
    
difference = 2 * epsilon     # variable to keep track of distance between v and v_prime

l = 0 # running index

[value_function_n,policy_index_n] = value_function_iteration(alpha,beta,f,Kapital_Grid,epsilon)

value_function_a = value_function_analytical(alpha_0,alpha_1,Kapital_Grid)

policy_function_a = policy_function_analytical(alpha,beta,A,Kapital_Grid)

policy_function_n = policy_function_from_policy_index(policy_index_n,Kapital_Grid)

optimal_path = find_optimal_path(k_0,n,policy_index_n,Kapital_Grid)
 
plt.plot(Kapital_Grid,value_function_n,'-',color = 'b')
plt.plot(Kapital_Grid,value_function_a,'--',color = 'r')
plt.plot(Kapital_Grid,policy_function_n,'-',color = 'b')
plt.plot(Kapital_Grid,policy_function_a,'--',color = 'r')
plt.plot(list(range(0,n)),optimal_path)
plt.show()


print("Done")
    

    
   
   
        