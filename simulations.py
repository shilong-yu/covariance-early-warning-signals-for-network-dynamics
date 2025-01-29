import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import multiprocessing
import pandas as pd

# df = pd.read_csv('nets\\barabasi_albert.csv')
# edge_list = list(df.itertuples(index=False, name=None))
# N = 100
# df2 = pd.read_csv('nets\\bats.csv')
# edge_list = list(df2[["from", "to"]].itertuples(index=False, name=None))
# N = 43
# df3 = pd.read_csv('nets\\nestbox.csv')
# edge_list = list(df3[["from", "to"]].itertuples(index=False, name=None))
# N = 126
# df4 = pd.read_csv('nets\\elephantseals.csv')
# edge_list = list(df4[["from", "to"]].itertuples(index=False, name=None))
# N = 46
# df5 = pd.read_csv('nets\\drugusers.csv')
# edge_list = list(df5[["from", "to"]].itertuples(index=False, name=None))
# N = 193
# df6 = pd.read_csv('nets\\highschoolboys.csv')
# edge_list = list(df6.itertuples(index=False, name=None))
# N = 70
# df7 = pd.read_csv('nets\\lizards.csv')
# edge_list = list(df7[["from", "to"]].itertuples(index=False, name=None))
# N = 60
# df8 = pd.read_csv('nets\\housefinches.csv')
# edge_list = list(df8[["from", "to"]].itertuples(index=False, name=None))
# N = 108
# df9 = pd.read_csv('nets\\LFR.csv')
# edge_list = list(df9.itertuples(index=False, name=None))
# edge_list = [(int(u[1:]), int(v[1:])) for u, v in edge_list]
# N = 100
# df10 = pd.read_csv('nets\\fitness.csv')
# edge_list = list(df10.itertuples(index=False, name=None))
# N = 87
# df11 = pd.read_csv('nets\\jazz.csv')
# edge_list = list(df11.itertuples(index=False, name=None))
# N = 198
# df12 = pd.read_csv('nets\\erdos_renyi.csv')
# edge_list = list(df12.itertuples(index=False, name=None))
# N = 100
# df13 = pd.read_csv('nets\\powerlaw.csv')
# edge_list = list(df13.itertuples(index=False, name=None))
# N = 100
# df14 = pd.read_csv('nets\\karate.csv')
# edge_list = list(df14.itertuples(index=False, name=None))
# N = 34
# df15 = pd.read_csv('nets\\surfers.csv')
# edge_list = list(df15[["from", "to"]].itertuples(index=False, name=None))
# N = 43
# df16 = pd.read_csv('nets\\netsci.csv')
# edge_list = list(df16[["from", "to"]].itertuples(index=False, name=None))
# N = 379
# nodes = set([node for edge in edge_list for node in edge])  # Get unique nodes
# node_mapping = {node: idx + 1 for idx, node in enumerate(nodes)}  # Map nodes to integers starting from 1

# Relabel edges using the mapping
# relabelled_edge_list = [(node_mapping[u], node_mapping[v]) for u, v in edge_list]
# df17 = pd.read_csv('nets\\tortoises.csv')
# edge_list = list(df17.itertuples(index=False, name=None))
# N = 94
# df18 = pd.read_csv('nets\\hall.csv')
# edge_list = list(df18[["from", "to"]].itertuples(index=False, name=None))
# N = 217
# df19 = pd.read_csv('nets\\voles.csv')
# edge_list = list(df19[["from", "to"]].itertuples(index=False, name=None))
# N = 111
# df20 = pd.read_csv('nets\\weaverbirds.csv')
# edge_list = list(df20.itertuples(index=False, name=None))
# N = 64
# df21 = pd.read_csv('nets\\pira.csv')
# edge_list = list(df21.itertuples(index=False, name=None))
# N = 128

# nodes = set([node for edge in edge_list for node in edge])  # Get unique nodes
# node_mapping = {node: idx + 1 for idx, node in enumerate(nodes)}  # Map nodes to integers starting from 1

# Relabel edges using the mapping
# relabelled_edge_list = [(node_mapping[u], node_mapping[v]) for u, v in edge_list]
# df22 = pd.read_csv('nets\\er_islands.csv')
# edge_list = list(df22.itertuples(index=False, name=None))
# N = 99
df23 = pd.read_csv('nets\\dolphins.csv')
edge_list = list(df23.itertuples(index=False, name=None))
N = 62

nodes = set([node for edge in edge_list for node in edge])  # Get unique nodes
node_mapping = {node: idx + 1 for idx, node in enumerate(nodes)}  # Map nodes to integers starting from 1

# Relabel edges using the mapping
relabelled_edge_list = [(node_mapping[u], node_mapping[v]) for u, v in edge_list]

start_time = time.time()
dt = 0.01  # size of time step
I = 20000  # number of time steps per simulation
L = 100  # number of samples for each node in order to create the covariance matrix
k = 200  # number of times we linearly increase u
# edges = np.array(edge_list)
edges = np.array(relabelled_edge_list)
# m = 2
# G = nx.barabasi_albert_graph(N, m)
# edge_list = G.edges()
# print(edge_list)


def double_well_coupled(x, noise, stress, D):
    # parameters in the double-well dynamics
    r1, r2, r3 = 1, 3, 5
    # u is the parameter
    dx = (-(x - r1) * (x - r2) * (x - r3) + stress) * dt + noise * math.sqrt(dt)
    contribution = np.zeros_like(x)

    # Create separate arrays for the indices from the edge list
    # edges = np.array(edge_list) cost a lot of time!
    i_indices = edges[:, 0] - 1
    j_indices = edges[:, 1] - 1
    # Vectorized contribution from edges
    # contribution[i_indices] += D * x[j_indices] * dt
    # contribution[j_indices] += D * x[i_indices] * dt
    np.add.at(contribution, i_indices, D * x[j_indices] * dt)
    np.add.at(contribution, j_indices, D * x[i_indices] * dt)

    dx += contribution
    nextx = x + dx
    return nextx


def mutualistic_interaction(x, noise, stress, D):
    # parameters in the mutualistic interaction dynamics
    C, Di, E, H, K = 1, 5, 0.9, 0.1, 5
    dx = (0.1 + stress + x * (1 - x / K) * (x / C - 1)) * dt + noise * np.sqrt(dt)

    contribution = np.zeros_like(x)
    # Extract i and j indices from edge list
    i_indices = edges[:, 0] - 1
    j_indices = edges[:, 1] - 1

    # Compute asymmetric interaction terms
    interaction_term_i = D * x[i_indices] * x[j_indices] / (Di + E * x[i_indices] + H * x[j_indices]) * dt
    interaction_term_j = D * x[i_indices] * x[j_indices] / (Di + E * x[j_indices] + H * x[i_indices]) * dt

    # Accumulate interactions into dx for both i and j nodes
    np.add.at(contribution, i_indices, interaction_term_i)  # Update dx[i]
    np.add.at(contribution, j_indices, interaction_term_j)  # Update dx[j]
    # contribution[i_indices] += interaction_term_i
    # contribution[j_indices] += interaction_term_j

    # Update x
    dx += contribution
    nextx = x + dx
    nextx[nextx <= 0] = 0
    return nextx


def gene_reg(x, noise, stress, D):
    # parameters in the gene regulatory dynamics
    B_gr, D_gr, h, f = 1, 1, 2, 1

    dx = (-B_gr * x + stress) * dt + noise * np.sqrt(dt)

    contribution = np.zeros_like(x)
    i_indices = edges[:, 0] - 1
    j_indices = edges[:, 1] - 1

    interaction_term_i = D * x[j_indices] * x[j_indices] * dt / (1 + x[j_indices] * x[j_indices])
    interaction_term_j = D * x[i_indices] * x[i_indices] * dt / (1 + x[i_indices] * x[i_indices])

    np.add.at(contribution, i_indices, interaction_term_i)
    np.add.at(contribution, j_indices, interaction_term_j)

    dx += contribution
    nextx = x + dx
    nextx[nextx <= 0] = 0
    return nextx


def sis(x, noise, lamda):
    # parameters in the SIS dynamics
    mu = 1

    dx = -mu * x * dt + noise * np.sqrt(dt)

    contribution = np.zeros_like(x)
    i_indices = edges[:, 0] - 1
    j_indices = edges[:, 1] - 1

    interaction_term_i = lamda * ((1 - x[i_indices]) * x[j_indices]) * dt
    interaction_term_j = lamda * ((1 - x[j_indices]) * x[i_indices]) * dt

    np.add.at(contribution, i_indices, interaction_term_i)
    np.add.at(contribution, j_indices, interaction_term_j)

    dx += contribution
    nextx = x + dx
    nextx[nextx <= 0] = 0
    return nextx


def simulate(u, sigma, stress, x, dy_type):
    X_all = np.zeros((N, I))  # all points used to generate the SDEs
    for i in range(0, I):
        # calculate the noise vector at each time point of a SDE iteration
        rands = np.random.normal(0, 1, (N, 1))
        noise = sigma * rands

        # temporarily stores the values of each node at a time point. It is an Nx1 vector.
        if dy_type == 1:
            temp = double_well_coupled(x, noise, u + stress, 0.05)
        elif dy_type == 2:
            temp = mutualistic_interaction(x, noise, u + stress, 1)
        elif dy_type == 3:
            temp = gene_reg(x, noise, u + stress, 1)
        elif dy_type == 4:
            temp = sis(x, noise, u)

        # stores the node values at a time into X_all.
        X_all[:, i] = temp.flatten()
        x = temp
    return X_all


def generate_sample_matrix(X):
    """
    X: The matrix that stores all the points of each node at each time point in a simulation
    X should be N by I.
    Returns the last 100 points for each node in a simulation which forms the data in order
    to calculate the sample covariance matrix.
    L: number of samples per node
    """
    indices = np.arange(I - L * 100, I, 100)
    X_samples = X[:, indices]
    return X_samples[:N, :]


def check_crit(temper, crit, dy_type):
    if np.max(temper) >= 3 and dy_type == 1:
        return crit
    elif np.min(temper) < 0.1 and dy_type == 2:
        return crit
    elif np.min(temper) < 0.1 and dy_type == 3:
        return crit
    elif np.max(temper) >= 0.1 and dy_type == 4:
        return crit
    else:
        return 0


def worker_function(i, noise, stress, x, du, dy_type):
    # Access the N by L sample points for u = i*du
    u_crit, binary = 0, 0
    temp = generate_sample_matrix(simulate(i * du, noise, stress, x, dy_type))

    # Calculate the mean of each row
    x_equi = np.mean(temp, axis=1)

    # Check critical conditions
    temp_crit = check_crit(x_equi, i * du, dy_type)
    if temp_crit != 0 and binary == 0:
        u_crit, binary = i, 1

    # Calculate the covariance matrix for temp
    C = np.cov(temp)

    # Calculate the variance using the trace of the covariance matrix
    variance = np.trace(C) / N

    return x_equi, variance, u_crit, C


def equilibrium(noise, stress, x, k, du, dy_type):
    # Initialize variables
    x_equi, variances, C_ensemble, u_crit = np.zeros(N * k), np.zeros(k), [], 0

    # Create a pool of worker processes
    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    with multiprocessing.Pool(processes=1) as pool:
        results = pool.starmap(worker_function, [(i, noise, stress, x, du, dy_type) for i in range(k)])
        # print(results)

    # Process results
    for i, (x_equi_i, variance_i, u_crit_i, C_i) in enumerate(results):
        x_equi[i * N:(i + 1) * N] = x_equi_i
        variances[i] = variance_i
        C_ensemble.append(C_i)
        if u_crit_i != 0 and u_crit == 0:
            u_crit = u_crit_i

    return x_equi, variances, u_crit, C_ensemble


def simulate_d(D, sigma, x, dy_type):
    X_all = np.zeros((N, I))
    # stress = np.zeros((N, 1))

    for i in range(0, I):
        # calculate the noise vector at each time point of a SDE iteration
        rands = np.random.normal(0, 1, (N, 1))
        noise = sigma * rands
        # temporarily stores the values of each node at a time point. It is a Nx1 vector.
        if dy_type == 1:
            temp = double_well_coupled(x, noise, 0, D)
        elif dy_type == 2:
            temp = mutualistic_interaction(x, noise, 0, D)
        elif dy_type == 3:
            temp = gene_reg(x, noise, 0, D)
        # stores the node values at a time into X_all.
        X_all[:, i] = temp.flatten()
        x = temp

    return X_all


def worker_function_d(i, noise, x, dD, dy_type):
    # Access the N by L sample points for u = i*du
    D_crit, binary = 0, 0
    if dy_type == 1:
        temp = generate_sample_matrix(simulate_d(i * dD, noise, x, dy_type))
    elif dy_type == 2 or dy_type == 3:
        temp = generate_sample_matrix(simulate_d(1 + i * dD, noise, x, dy_type))

    # Calculate the mean of each row
    x_equi = np.mean(temp, axis=1)

    # Check critical conditions
    if dy_type == 1:
        temp_crit = check_crit(x_equi, i * dD, dy_type)
    elif dy_type == 2 or dy_type == 3:
        temp_crit = check_crit(x_equi, 1 + i * dD, dy_type)

    if temp_crit != 0 and binary == 0:
        D_crit, binary = i, 1

    # Calculate the covariance matrix for temp
    C = np.cov(temp)

    # Calculate the variance using the trace of the covariance matrix
    variance = np.trace(C) / N

    return x_equi, variance, D_crit, C


def equilibrium_d(noise, x, k, dD, dy_type):
    x_equi, variances, C_ensemble, D_crit = np.zeros(N * k), np.zeros(k), [], 0

    # Create a pool of worker processes
    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    with multiprocessing.Pool(processes=1) as pool:

        results = pool.starmap(worker_function_d, [(i, noise, x, dD, dy_type) for i in range(k)])
        # print(results)

    # Process results
    for i, (x_equi_i, variance_i, D_crit_i, C_i) in enumerate(results):
        x_equi[i * N:(i + 1) * N] = x_equi_i
        variances[i] = variance_i
        C_ensemble.append(C_i)
        if D_crit_i != 0 and D_crit == 0:
            D_crit = D_crit_i

    return x_equi, variances, D_crit, C_ensemble


def init_conditions(homo_hetero, dy_type):
    sigma = np.zeros((N, 1))  # vector of how big the noise is for each node
    stress = np.zeros((N, 1))  # stress vector which is also the bifurcation parameter
    x = np.zeros((N, 1))  # the initial starting states for the N nodes

    if dy_type == 1:
        x.fill(1)
        sigma_temp = 0.05
        du = 0.025
        dD = 0.0025
        Ds = np.linspace(0, (k-1) * dD, k)
    elif dy_type == 2:
        x.fill(5)
        sigma_temp = 0.25
        du = -0.1
        dD = -0.01
        Ds = np.linspace(1, (k-1) * dD + 1, k)
    elif dy_type == 3:
        x.fill(5)
        sigma_temp = 0.000005
        du = -0.01
        dD = -0.01
        Ds = np.linspace(1, (k-1) * dD + 1, k)
    elif dy_type == 4:
        x.fill(0.001)
        sigma_temp = 0.0005
        du = 0.0025
        dD = 0
        Ds = np.linspace(1, (k - 1) * dD + 1, k)

    if homo_hetero == 1:
        sigma.fill(sigma_temp)
    elif homo_hetero == 2:
        sigma = np.random.uniform(-0.9 * sigma_temp, 0.9 * sigma_temp, (N, 1))
        stress = np.random.uniform(-0.25, 0.25, (N, 1))
    elif homo_hetero == 3:
        sigma = np.random.uniform(-0.9 * sigma_temp, 0.9 * sigma_temp, (N, 1))

    us = np.linspace(0, (k - 1) * du, k)  # vector of u values
    return stress, sigma, x, du, us, dD, Ds


# stress, sigma, x, du, us = init_conditions(2)
# x_equis, var, u_crit, C_ensemble = equilibrium(sigma, stress, x)

def u_graph(us, x_equis, var, u_crit):
    fig, ax = plt.subplots(2)
    x_equis_reshaped = x_equis.reshape(k, N)
    for j in range(N):
        ax[0].plot(us, x_equis_reshaped[:, j])
    ax[1].plot(us, var)
    ax[0].set_title('x^*_i vs u values')
    ax[0].set_xlabel('u')
    ax[0].set_ylabel('x^*_i')
    ax[1].set_xlabel('u')
    ax[1].set_ylabel('average variance')
    ax[1].set_title('average variances using every node vs u values')
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    # plt.show()
    print('u_crit is ', u_crit)


def d_graph(Ds, x_equis, var, D_crit):
    fig, ax = plt.subplots(2)
    x_equis_reshaped = x_equis.reshape(k, N)
    for j in range(N):
        ax[0].plot(Ds, x_equis_reshaped[:, j])
    ax[1].plot(Ds, var)
    ax[0].set_title('x^*_i vs D values')
    ax[0].set_xlabel('D')
    ax[0].set_ylabel('x^*_i')
    ax[1].set_xlabel('D')
    ax[1].set_ylabel('average variance')
    ax[1].set_title('average variances using every node vs D values')
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    # plt.show()
    print('D_crit is ', D_crit)


# Note: Make sure to call equilibrium() only in the __main__ block to avoid issues with multiprocessing on Windows.
if __name__ == "__main__":
    start_time = time.time()
    filenames = []
    network_type = ['dolphins']
    dynamic_type = ['d', 'm', 'g', 's']
    noise_type = ['homo', 'hetero']
    bifur_type = ['u', 'D']
    for i in network_type:
        for j in dynamic_type:
            for nt in noise_type:
                for m in bifur_type:
                    if not(j == 's' and m == 'D'):
                        filenames.append(i + '-' + j + '-' + m + '-' + nt)

    print(filenames)
    counter, count = 0, 1
    for i in range(1, 5):
        for j in range(1, 3):
            stress, sigma, x, du, us, dD, Ds = init_conditions(j, i)
            if i == 4 and j == 2:
                stress, sigma, x , du, us, dD, Ds = init_conditions(3, 4)
            x_equis, var, u_crit, C_ensemble = equilibrium(sigma, stress, x, k, du, i)
            u_graph(us, x_equis, var, u_crit)
            with open(filenames[counter], 'w') as f:
                for C in C_ensemble:
                    np.savetxt(f, C, header="Matrix " + str(count))
                    count += 1
                np.savetxt(f, var, header="variances")
                np.savetxt(f, x_equis, header="x_equis")
                np.savetxt(f, [int(u_crit)], header="u_crit")
            counter += 1
            count = 1
            if i < 4:
                x_equis, var, D_crit, C_ensemble = equilibrium_d(sigma, x, k, dD, i)
                d_graph(Ds, x_equis, var, D_crit)
                with open(filenames[counter], 'w') as f:
                    for C in C_ensemble:
                        np.savetxt(f, C, header="Matrix " + str(count))
                        count += 1
                    np.savetxt(f, var, header="variances")
                    np.savetxt(f, x_equis, header="x_equis")
                    np.savetxt(f, [int(D_crit)], header="u_crit")
                counter += 1
                count = 1

    # C_1 = C_ensemble[round(0.1 * u_crit)]
    # C_2 = C_ensemble[round(0.9 * u_crit)]
    # with open("Double-well-u-BA-hetero", "w") as f:
        # np.savetxt(f, C_1, header="Matrix 1")
        # f.write("\n")
        # np.savetxt(f, C_2, header="Matrix 2")
        # f.write("\n")
        # np.savetxt(f, var, header="variances")

    end_time = time.time()
    print(end_time - start_time)

    plt.show()
