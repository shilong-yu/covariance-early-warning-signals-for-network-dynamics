# Covariance-early-warning-signals-for-network-dynamics

The important files to look at are simulations.py, 2025 final d and tau calculations.py, and 2025 ANOVA for dw.py. The other simulations files are simply variants of simulations.py for specific dynamics and direction of simulation, and the another ANOVA files are almost identical to 2025 ANOVA for dw.py

1. simulations.py

We use the Euler-Maruyama method, so the functions double_well_coupled(), mutualistic_interaction(), gene_reg(), and sis() update and return the new value of simulation from the previous time step. 

The functions simulate() and simulate_d() simulates completely one dynamics with a specific control parameter value.

generate_sample_matrix() generates the sample covariance matrix from one simulation.

equilibrium() and equilibrium() returns the value at which critical transition occurs as well as all the covariance matrices obtained up to that critical value.

2. 2025 final d and tau calculations

read_matrix() reads a sample covariance matrix

d() and d_calc() calculate the d with the given mean and variance at two points.

gen_mean_of_var() calculates the mean of the proposed EWS, and gen_var_of_var() calculates the variance of the proposed EWS.

tau_calc() calculates the tau value of an EWS

generate_arbi_entries() generates the set of entries selected to calculate the EWS.
   
3. 2025 ANOVA for dw

read_file() reads the d and tau values.

data_generation() prepares for ANOVA analyses.



 
