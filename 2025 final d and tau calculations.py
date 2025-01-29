import numpy as np
import random
import itertools
import scipy.stats as stats
import heapq
import matplotlib.pyplot as plt
import math
import numpy as np
import time
from itertools import chain


# Function to read a specific matrix from the file
def read_matrix(file_path, matrix_index):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    start_line = None
    end_line = None

    # Find the start and end lines of the required matrix
    for i, line in enumerate(lines):
        if line.startswith('# Matrix ' + str(matrix_index)):
            start_line = i + 1  # Start from the line after the label
        elif start_line is not None and line.startswith('#'):
            end_line = i
            break

    # Extract matrix data from the identified range of lines
    matrix_data = lines[start_line:end_line]
    matrix = np.loadtxt(matrix_data)
    return matrix

Z = 100
n = 5
# u critical roughly at ...
# bifur1 u value that is far from critical transition: 10% of u_critical
# bifur2 u value that is close to critical transition: 90% of u_critical
# du = -0.025
dD = -0.0025
kk = 200
# us = np.linspace(0, (kk - 1) * du, kk)
Ds = np.linspace(0.1, (kk - 1) * dD + 0.1, kk)
L = 100


def d(mu1, mu2, var1, var2):
    return abs(mu2 - mu1) / np.sqrt(var1 + var2)


def gen_mean_of_var(X, k, A):
    """
    X: The Covariance Matrix
    k: The number of points sampled:
    A: a list of coordinates (x1, y1, x2, y2, ...) of the covariance matrix that we will use
    The list should have 2k entries.
    """
    temp = 0
    for i in range(0, k * 2):
        if i % 2 == 0:
            temp += X[A[i]][A[i + 1]]
    return 1 / k * temp


def gen_var_of_var(X, k, A):
    temp = 0
    for i in range(0, k):
        for j in range(0, k):
            temp += X[A[i * 2]][A[j * 2]] * X[A[i * 2 + 1]][A[j * 2 + 1]] + \
                    X[A[i * 2]][A[j * 2 + 1]] * X[A[j * 2]][A[i * 2 + 1]]
    return 1 / (k * k * (L - 1)) * temp


def d_calc(X, C1, C2):
    mu1 = gen_mean_of_var(C1, int(len(X) / 2), X)
    mu2 = gen_mean_of_var(C2, int(len(X) / 2), X)
    var1 = gen_var_of_var(C1, int(len(X) / 2), X)
    var2 = gen_var_of_var(C2, int(len(X) / 2), X)
    d_val = d(mu1, mu2, var1, var2)
    return d_val


def tau_calc(Y, C_ensembles, u_crit):
    avg_vars = []
    for Cov in C_ensembles:
        avg_var = gen_mean_of_var(Cov, int(len(Y)/2), Y)
        avg_vars.append(avg_var)
    # tau, p_value = stats.kendalltau(us[0:u_crit], avg_vars[0:u_crit])
    tau, p_value = stats.kendalltau(Ds[0:u_crit], avg_vars[0:u_crit])
    return tau


step = np.linspace(0, Z - 1, Z)


def generate_arbi_entries(n, N):
    AAs, BBs, CCs, DDs = [], [], [], []
    # AA on diagonals
    # BB diagonal or upper triangular random
    # CC minor matrices
    # DD same rows
    count = 0
    diagonal_indices = [(i, i) for i in range(N)]
    while count < 100:
        # AA
        selected_indices_A = random.sample(diagonal_indices, n)
        if set(selected_indices_A) not in [set(lst) for lst in AAs]:
            AAs.append(selected_indices_A)
            count += 1
    count = 0
    upper_triangular_indices = [(i, j) for i in range(N) for j in range(i, N)]
    while count < 100:
        # BB
        selected_indices_B = random.sample(upper_triangular_indices, n)
        if set(selected_indices_B) not in [set(lst) for lst in BBs]:
            BBs.append(selected_indices_B)
            count += 1
    count = 0
    indices = [i for i in range(N)]
    while count < 100:
        # CC
        selected_indices_C = []
        selected_indices = random.sample(indices, n)
        for j in selected_indices:
            for k in selected_indices:
                selected_indices_C.append((j, k))
        if set(selected_indices_C) not in [set(lst) for lst in CCs]:
            CCs.append(selected_indices_C)
            count += 1
    count = 0
    indices = [i for i in range(N)]
    while count < 100:
        selected_indices_D = []
        selected_indices = random.sample(indices, 1)
        selected_indices_2 = random.sample(indices, n)
        for i in selected_indices:
            for j in selected_indices_2:
                selected_indices_D.append((i, j))
        if set(selected_indices_D) not in [set(lst) for lst in DDs]:
            DDs.append(selected_indices_D)
            count += 1

        # BB = list(chain.from_iterable(selected_indices))
    return AAs, BBs, CCs, DDs


def calculate_d_and_tau(Z, C1, C2, C_ensembles, u_crit, N):
    d_diags, d_offs, tau_diags, tau_offs, d_minors, d_rows, tau_minors, tau_rows = [], [], [], [], [], [], [], []
    AAs, BBs, CCs, DDs = generate_arbi_entries(n, N)

    for i in range(0, Z):
        # calculate d values and append them
        AA = [item for sublist in AAs[i] for item in sublist]
        BB = [item for sublist in BBs[i] for item in sublist]
        CC = [item for sublist in CCs[i] for item in sublist]
        DD = [item for sublist in DDs[i] for item in sublist]

        d_diag = d_calc(AA, C1, C2)
        d_off_diag = d_calc(BB, C1, C2)
        d_diags.append(d_diag)
        d_offs.append(d_off_diag)
        d_minor = d_calc(CC, C1, C2)
        d_minors.append(d_minor)
        d_row = d_calc(DD, C1, C2)
        d_rows.append(d_row)

        # calculate tau values and append them
        tau_diag = tau_calc(AA, C_ensembles, u_crit)
        tau_off = tau_calc(BB, C_ensembles, u_crit)
        tau_diags.append(tau_diag)
        tau_offs.append(tau_off)
        tau_minor = tau_calc(CC, C_ensembles, u_crit)
        tau_row = tau_calc(DD, C_ensembles, u_crit)
        tau_minors.append(tau_minor)
        tau_rows.append(tau_row)

    return d_diags, d_offs, tau_diags, tau_offs, d_minors, d_rows, tau_minors, tau_rows


def draw(step, d_diags, d_offs, d_minors, d_rows, tau_diags, tau_offs, tau_minors, tau_rows):
    # draw the d values for random off-diagonal and on-diagonal selections
    # plt.scatter(step, d_diags, edgecolors='r', facecolors='none', label='on diag')
    # plt.scatter(step, d_offs, edgecolors='b', facecolors='none', label='off diag')
    # plt.scatter(step, d_minors, edgecolors='g', facecolors='none', label='minors')
    # plt.scatter(step, d_rows, edgecolors='y', facecolors='none', label='rows')
    # plt.ylabel('d values')
    # plt.xlabel('iterations')
    # plt.title('d values for randomized off-diagonal, on-diagonal, minor and row selections')
    # plt.legend(loc='lower right')
    print('The average of d values for random on-diagonal selections is ', np.mean(np.array(d_diags)))
    print('The average of d values for random off-diagonal selections is ', np.mean(np.array(d_offs)))
    print('The average of d values for random minor selections is ', np.mean(np.array(d_minors)))
    print('The average of d values for random row selections is ', np.mean(np.array(d_rows)))
    print('The average of tau values for random on-diagonal selections is ', np.mean(np.array(tau_diags)))
    print('The average of tau values for random off-diagonal selections is ', np.mean(np.array(tau_offs)))
    print('The average of tau values for random minor selections is ', np.mean(np.array(tau_minors)))
    print('The average of tau values for random row selections is ', np.mean(np.array(tau_rows)))
    """
    # plt.show()
    # draw the d vs tau values for random off-diagonal and on-diagonal selection
    plt.scatter(tau_diags, d_diags, edgecolors='r', facecolors='none', label='on-diagonal')
    plt.scatter(tau_rows, d_rows, edgecolors='y', facecolors='none', label='row')
    plt.scatter(tau_minors, d_minors, edgecolors='g', facecolors='none', label='minor')
    plt.scatter(tau_offs, d_offs, edgecolors='b', facecolors='none', label='random')

    plt.title('d vs τ for random minor, off-diagonal, on-diagonal, and row selections')
    plt.ylabel('d', fontsize=10 * 1.5, fontstyle='italic')
    plt.xlabel('τ', fontsize=10 * 1.5, fontstyle='italic')
    plt.legend(loc='upper left', fontsize=10 * 1.5)
    plt.xticks(fontsize=10 * 1.5)
    plt.yticks(fontsize=10 * 1.5)
    plt.show()
    """


# d_diags, d_offs, tau_diags, tau_offs, d_minors, d_rows, tau_minors, tau_rows = calculate_d_and_tau(Z)
# draw(step, d_diags, d_offs, d_minors, d_rows, tau_diags, tau_offs, tau_minors, tau_rows)

if __name__ == "__main__":
    start_time = time.time()
    C_ensemble = []
    Ns = [100, 100, 43, 43, 62, 62, 193, 193, 46, 46, 99, 99, 100, 100, 87, 87, 217, 217, 70, 70, 108, 108,
        198, 198, 34, 34, 100, 100, 60, 60, 126, 126, 379, 379, 128, 128, 100, 100, 43, 43, 94, 94, 111, 111, 64, 64]
    u_crits = [52, 52, 48, 48, 62, 62, 57, 57, 68, 69, 66, 66, 63, 64, 46, 46, 45, 45,
               54, 54, 45, 45, 43, 43, 55, 55, 52, 52, 52, 52, 45, 45, 48, 48, 50, 50,
               45, 45, 49, 49, 59, 59, 64, 64, 52, 53]

    filenames = ['barabasi_albert-d-D-homo-down', 'barabasi_albert-d-D-hetero-down',
                     'bats-d-D-homo-down', 'bats-d-D-hetero-down',
                     'dolphins-d-D-homo-down', 'dolphins-d-D-hetero-down',
                     'drugusers-d-D-homo-down', 'drugusers-d-D-hetero-down',
                     'elephantseals-d-D-homo-down', 'elephantseals-d-D-hetero-down',
                     'er_islands-d-D-homo-down', 'er_islands-d-D-hetero-down',
                     'erdos_renyi-d-D-homo-down', 'erdos_renyi-d-D-hetero-down',
                     'fitness-d-D-homo-down', 'fitness-d-D-hetero-down',
                     'hall-d-D-homo-down', 'hall-d-D-hetero-down',
                     'highschoolboys-d-D-homo-down', 'highschoolboys-d-D-hetero-down',
                     'housefinches-d-D-homo-down', 'housefinches-d-D-hetero-down',
                     'jazz-d-D-homo-down', 'jazz-d-D-hetero-down',
                     'karate-d-D-homo-down', 'karate-d-D-hetero-down',
                     'LFR-d-D-homo-down', 'LFR-d-D-hetero-down',
                     'lizards-d-D-homo-down', 'lizards-d-D-hetero-down',
                     'nestbox-d-D-homo-down', 'nestbox-d-D-hetero-down',
                     'netsci-d-D-homo-down', 'netsci-d-D-hetero-down',
                     'pira-d-D-homo-down', 'pira-d-D-hetero-down',
                     'powerlaw-d-D-homo-down', 'powerlaw-d-D-hetero-down',
                     'surfers-d-D-homo-down', 'surfers-d-D-hetero-down',
                     'tortoises-d-D-homo-down', 'tortoises-d-D-hetero-down',
                     'voles-d-D-homo-down', 'voles-d-D-hetero-down',
                     'weaverbirds-d-D-homo-down', 'weaverbirds-d-D-hetero-down']

    fileoutputs = ['barabasi_albert-d-D-homo-down-2026data', 'barabasi_albert-d-D-hetero-down-2026data',
                 'bats-d-D-homo-down-2026data', 'bats-d-D-hetero-down-2026data',
                 'dolphins-d-D-homo-down-2026data', 'dolphins-d-D-hetero-down-2026data',
                 'drugusers-d-D-homo-down-2026data', 'drugusers-d-D-hetero-down-2026data',
                 'elephantseals-d-D-homo-down-2026data', 'elephantseals-d-D-hetero-down-2026data',
                 'er_islands-d-D-homo-down-2026data', 'er_islands-d-D-hetero-down-2026data',
                 'erdos_renyi-d-D-homo-down-2026data', 'erdos_renyi-d-D-hetero-down-2026data',
                 'fitness-d-D-homo-down-2026data', 'fitness-d-D-hetero-down-2026data',
                 'hall-d-D-homo-down-2026data', 'hall-d-D-hetero-down-2026data',
                 'highschoolboys-d-D-homo-down-2026data', 'highschoolboys-d-D-hetero-down-2026data',
                 'housefinches-d-D-homo-down-2026data', 'housefinches-d-D-hetero-down-2026data',
                   'jazz-d-D-homo-down-2026data', 'jazz-d-D-hetero-down-2026data',
                   'karate-d-D-homo-down-2026data', 'karate-d-D-hetero-down-2026data',
                   'LFR-d-D-homo-down-2026data', 'LFR-d-D-hetero-down-2026data',
                   'lizards-d-D-homo-down-2026data', 'lizards-d-D-hetero-down-2026data',
                   'nestbox-d-D-homo-down-2026data', 'nestbox-d-D-hetero-down-2026data',
                   'netsci-d-D-homo-down-2026data', 'netsci-d-D-hetero-down-2026data',
                   'pira-d-D-homo-down-2026data', 'pira-d-D-hetero-down-2026data',
                   'powerlaw-d-D-homo-down-2026data', 'powerlaw-d-D-hetero-down-2026data',
                   'surfers-d-D-homo-down-2026data', 'surfers-d-D-hetero-down-2026data',
                   'tortoises-d-D-homo-down-2026data', 'tortoises-d-D-hetero-down-2026data',
                   'voles-d-D-homo-down-2026data', 'voles-d-D-hetero-down-2026data',
                   'weaverbirds-d-D-homo-down-2026data', 'weaverbirds-d-D-hetero-down-2026data']

    counter = 0
    for file in filenames:
        for i in range(1, u_crits[counter] + 5):
            C_ensemble.append(read_matrix(file, i))
        bifur1 = round(0.1*u_crits[counter])
        bifur2 = round(0.9*u_crits[counter])
        C1 = C_ensemble[bifur1]
        C2 = C_ensemble[bifur2]
        d_diags, d_offs, tau_diags, tau_offs, d_minors, d_rows, tau_minors, \
            tau_rows = calculate_d_and_tau(Z, C1, C2, C_ensemble, u_crits[counter], Ns[counter])
        C_ensemble = []
        draw(step, d_diags, d_offs, d_minors, d_rows, tau_diags, tau_offs, tau_minors, tau_rows)
        with open(fileoutputs[counter], 'w') as f:
            np.savetxt(f, d_diags, header="d_diags")
            np.savetxt(f, d_offs, header="d_offs")
            np.savetxt(f, d_minors, header="d_minors")
            np.savetxt(f, d_rows, header="d_rows")
            np.savetxt(f, tau_diags, header="tau_diags")
            np.savetxt(f, tau_offs, header="tau_offs")
            np.savetxt(f, tau_minors, header="tau_minors")
            np.savetxt(f, tau_rows, header="tau_rows")
        counter += 1
    end_time = time.time()
    print(end_time - start_time)