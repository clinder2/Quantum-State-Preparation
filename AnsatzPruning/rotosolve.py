import numpy as np
from qiskit.primitives import StatevectorEstimator as Estimator

def rotosolve(circuit, num_steps, hamiltonian):
    params_dict = {}
    for param in circuit.parameters: 
        params_dict[param] = 1; 

    for step in range(num_steps): 
        # print(step, params_dict.values())
        for key, value in params_dict.items(): 
            phi = value

            #copies of params to measure expectation 
            params_phi      = params_dict.copy()
            params_phi_plus = params_dict.copy()
            params_phi_minus= params_dict.copy()

            params_phi_plus[key] = phi  + np.pi/2
            params_phi_minus[key] = phi  - np.pi/2


            #get expectation 
            exp_phi = costfunc(circuit, params_phi, hamiltonian)
            exp_phi_plus = costfunc(circuit, params_phi_plus, hamiltonian)
            exp_phi_minus =costfunc(circuit, params_phi_minus, hamiltonian)

            #assuming k = 0 in rotosolve step
            theta_d = phi - np.pi/2 - np.arctan2(2 * exp_phi - exp_phi_plus - exp_phi_minus, exp_phi_plus - exp_phi_minus)
            theta_d = ((theta_d + np.pi) % (2*np.pi)) - np.pi            
            params_dict[key] = theta_d

    # rebind new parameter values    
    for param in circuit.parameters: 
        param.assign(param, params_dict[param])
    
    opt_params = list(params_dict.values())
    print(opt_params)
    return opt_params
#reassigns params and gets expectation 
def costfunc(circuit, params, observable): 

    #bind params to cirucit: 
    param_list = []
    bound_circuit = circuit.copy()
    for param in bound_circuit.parameters: 
        param.assign(param, params[param])
        param_list.append(params[param])

    estimator = Estimator()
    pub = (bound_circuit, observable, param_list)
    cost = estimator.run([pub]).result()[0].data.evs

    return cost
