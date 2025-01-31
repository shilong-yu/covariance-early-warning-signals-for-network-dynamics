import random
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

df15 = pd.read_csv('nets\\surfers.csv')
edge_list = list(df15[["from", "to"]].itertuples(index=False, name=None))
N = 43

# df7 = pd.read_csv('nets\\lizards.csv')
# edge_list = list(df7[["from", "to"]].itertuples(index=False, name=None))
# N = 60
# print(edge_list)

"""
df16 = pd.read_csv('nets\\netsci.csv')
edge_list = list(df16[["from", "to"]].itertuples(index=False, name=None))
N = 379
nodes = set([node for edge in edge_list for node in edge])  # Get unique nodes
node_mapping = {node: idx + 1 for idx, node in enumerate(nodes)}  # Map nodes to integers starting from 1

# Relabel edges using the mapping
temp_edge_list = [(node_mapping[u], node_mapping[v]) for u, v in edge_list]
edge_list = temp_edge_list
"""


def covariance_to_correlation(cov_matrix):
    GG = []
    std_devs = np.sqrt(np.diag(cov_matrix))

    stddev_outer = np.outer(std_devs, std_devs)

    correlation_matrix = cov_matrix / stddev_outer
    masked_matrix = correlation_matrix.copy()
    np.fill_diagonal(masked_matrix, -np.inf)

    upper_tri_indices = np.triu_indices(N, k=1)  # k=1 excludes diagonal
    upper_tri_values = masked_matrix[upper_tri_indices]

    largest_indices_sorted = np.argsort(upper_tri_values)[-5:]  # Indices of the five largest elements
    result_indices = [(upper_tri_indices[0][i], upper_tri_indices[1][i]) for i in largest_indices_sorted]

    for i in range(5):
        GG.append(result_indices[i][0])
        GG.append(result_indices[i][1])
    return GG


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
L = 100
k = 250
# u critical roughly at ...
# bifur1 u value that is far from critical transition: 10% of u_critical
# bifur2 u value that is close to critical transition: 90% of u_critical

# Double well going up D
dD = 0.0025
Ds = np.linspace(0, (k-1)*dD, k)

# Double well going down D
# dD = -0.0025
# Ds = np.linspace(0, (k - 1) * dD, k)

# Mutualistic going down D
# dD = -0.01
# Ds = np.linspace(1, (k-1)*dD + 1, k)


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
        avg_var = gen_mean_of_var(Cov, int(len(Y) / 2), Y)
        avg_vars.append(avg_var)
    # tau, p_value = stats.kendalltau(us[0:u_crit], avg_vars[0:u_crit])
    tau, p_value = stats.kendalltau(Ds[0:u_crit], avg_vars[0:u_crit])
    return tau


step = np.linspace(0, Z - 1, Z)


def generate_arbi_entries(n, N):
    AAs, BBs, CCs, DDs, EEs, FFs = [], [], [], [], [], []
    # AA diagonal
    # BB random
    # CC minor matrices
    # DD same row
    # EE upper triangular minor
    # FF edge
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
    count = 0
    indices = [i for i in range(N)]
    while count < 100:
        selected_indices_E = []
        selected_indices = random.sample(indices, n)
        for i in selected_indices:
            for j in selected_indices:
                if j > i:
                    selected_indices_E.append((i, j))
        if set(selected_indices_E) not in [set(lst) for lst in EEs]:
            EEs.append(selected_indices_E)
            count += 1
    count = 0
    while count < 100:
        selected_indices_F = []
        selected_indices = random.sample(edge_list, n)
        # print(selected_indices)
        for i in selected_indices:
            selected_indices_F.append((i[0] - 1, i[1] - 1))
        if set(selected_indices_F) not in [set(lst) for lst in FFs]:
            FFs.append(selected_indices_F)
            count += 1

    return AAs, BBs, CCs, DDs, EEs, FFs


def calculate_d_and_tau(Z, C1, C2, C_ensembles, u_crit, N):
    d_diags, d_offs, tau_diags, tau_offs, d_minors, d_rows, tau_minors, tau_rows = [], [], [], [], [], [], [], []
    d_uppers, tau_uppers, d_edges, tau_edges = [], [], [], []
    d_corr, tau_corr = 0, 0
    AAs, BBs, CCs, DDs, EEs, FFs = generate_arbi_entries(n, N)
    for i in range(0, Z):
        AA = [item for sublist in AAs[i] for item in sublist]
        BB = [item for sublist in BBs[i] for item in sublist]
        CC = [item for sublist in CCs[i] for item in sublist]
        DD = [item for sublist in DDs[i] for item in sublist]
        EE = [item for sublist in EEs[i] for item in sublist]
        FF = [item for sublist in FFs[i] for item in sublist]

        # calculate d values and append them
        d_diag = d_calc(AA, C1, C2)
        d_off_diag = d_calc(BB, C1, C2)
        d_diags.append(d_diag)
        d_offs.append(d_off_diag)
        d_minor = d_calc(CC, C1, C2)
        d_minors.append(d_minor)
        d_row = d_calc(DD, C1, C2)
        d_rows.append(d_row)
        d_upper = d_calc(EE, C1, C2)
        d_uppers.append(d_upper)
        d_edge = d_calc(FF, C1, C2)
        d_edges.append(d_edge)

        # calculate tau values and append them
        tau_diag = tau_calc(AA, C_ensembles, u_crit)
        tau_off = tau_calc(BB, C_ensembles, u_crit)
        tau_diags.append(tau_diag)
        tau_offs.append(tau_off)
        tau_minor = tau_calc(CC, C_ensembles, u_crit)
        tau_row = tau_calc(DD, C_ensembles, u_crit)
        tau_minors.append(tau_minor)
        tau_rows.append(tau_row)
        tau_upper = tau_calc(EE, C_ensembles, u_crit)
        tau_uppers.append(tau_upper)
        tau_edge = tau_calc(FF, C_ensembles, u_crit)
        tau_edges.append(tau_edge)
    GG = covariance_to_correlation(C_ensembles[0])
    print(GG)
    d_corr = d_calc(GG, C1, C2)
    tau_corr = tau_calc(GG, C_ensembles, u_crit)

    return d_diags, d_offs, tau_diags, tau_offs, d_minors, d_rows, tau_minors, tau_rows, \
        d_uppers, tau_uppers, d_edges, tau_edges, d_corr, tau_corr


def draw(step, d_diags, d_offs, d_minors, d_rows, d_uppers, d_edges, d_corr, tau_diags, tau_offs, tau_minors, tau_rows,
         tau_uppers, tau_edges, tau_corr):
    # draw the d values for random off-diagonal and on-diagonal selections
    plt.scatter(step, d_diags, edgecolors='r', facecolors='none', label='on diag')
    plt.scatter(step, d_offs, edgecolors='b', facecolors='none', label='off diag')
    plt.scatter(step, d_minors, edgecolors='g', facecolors='none', label='minors')
    plt.scatter(step, d_rows, edgecolors='y', facecolors='none', label='rows')
    plt.ylabel('d values')
    plt.xlabel('iterations')
    plt.title('d values for randomized off-diagonal, on-diagonal, minor and row selections')
    plt.legend(loc='lower right')
    print('The average of d values for random on-diagonal selections is ', np.mean(np.array(d_diags)))
    print('The average of d values for random off-diagonal selections is ', np.mean(np.array(d_offs)))
    print('The average of d values for random minor selections is ', np.mean(np.array(d_minors)))
    print('The average of d values for random row selections is ', np.mean(np.array(d_rows)))
    plt.show()

    # draw the d vs tau values for random off-diagonal and on-diagonal selection
    plt.scatter(tau_diags, d_diags, edgecolors='r', facecolors='none', label='on-diagonal')
    plt.scatter(tau_rows, d_rows, edgecolors='y', facecolors='none', label='row')
    plt.scatter(tau_minors, d_minors, edgecolors='g', facecolors='none', label='minor')
    plt.scatter(tau_offs, d_offs, edgecolors='b', facecolors='none', label='random')
    plt.scatter(tau_uppers, d_uppers, edgecolors='m', facecolors='none', label='upper')
    plt.scatter(tau_edges, d_edges, edgecolors='k', facecolors='none', label='edges')
    plt.scatter(tau_corr, d_corr, marker='x', facecolors='k', label='corr')
    # plt.title('d vs τ for random minor, off-diagonal, on-diagonal, and row selections')
    plt.ylabel('d', fontsize=10 * 1.5, fontstyle='italic')
    plt.xlabel('τ', fontsize=10 * 1.5, fontstyle='italic')
    # plt.legend(loc='upper right', fontsize=10 * 1.5)
    plt.legend(loc='upper right', fontsize=10)
    plt.xticks(fontsize=10 * 1.5)
    plt.yticks(fontsize=10 * 1.5)
    plt.show()


if __name__ == "__main__":
    start_time = time.time()
    counter = 0
    C_ensemble = []

    # filenames = ['lizards-d-D-homo-down']
    # u_crits = [52]
    # Ns = [60]
    # fileoutputs = ['lizards-d-D-homo-down-3000']

    # filenames = ['netsci-m-D-hetero']
    # u_crits = [105]
    # Ns = [379]
    # fileoutputs = ['netsci-m-D-hetero-3000']

    filenames = ['surfers-d-D-hetero']
    u_crits = [31]
    Ns = [43]
    fileoutputs = ['surfers-d-D-hetero-3000']

    for file in filenames:
        for i in range(1, 50):
            C_ensemble.append(read_matrix(file, i))
        bifur1 = round(0.1 * u_crits[counter])
        bifur2 = round(0.9 * u_crits[counter])
        C1 = C_ensemble[bifur1]
        C2 = C_ensemble[bifur2]
        d_diags, d_offs, tau_diags, tau_offs, d_minors, d_rows, tau_minors, \
            tau_rows, d_uppers, tau_uppers, d_edges, tau_edges, d_corr, tau_corr \
            = calculate_d_and_tau(Z, C1, C2, C_ensemble, u_crits[counter], Ns[counter])
        C_ensemble = []
        draw(step, d_diags, d_offs, d_minors, d_rows, d_uppers, d_edges, d_corr, tau_diags, tau_offs, tau_minors,
             tau_rows, tau_uppers, tau_edges, tau_corr)
        with open(fileoutputs[0], 'w') as f:
            np.savetxt(f, d_diags, header="d_diags")
            np.savetxt(f, d_offs, header="d_offs")
            np.savetxt(f, d_minors, header="d_minors")
            np.savetxt(f, d_rows, header="d_rows")
            np.savetxt(f, tau_diags, header="tau_diags")
            np.savetxt(f, tau_offs, header="tau_offs")
            np.savetxt(f, tau_minors, header="tau_minors")
            np.savetxt(f, tau_rows, header="tau_rows")
            np.savetxt(f, d_uppers, header="d_uppers")
            np.savetxt(f, d_edges, header="d_edges")
            np.savetxt(f, tau_uppers, header="tau_uppers")
            np.savetxt(f, tau_edges, header="tau_edges")
            np.savetxt(f, [d_corr], header="d_corr")
            np.savetxt(f, [tau_corr], header="tau_corr")
    C_ensemble = []
    end_time = time.time()
    print(end_time - start_time)
