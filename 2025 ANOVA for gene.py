import numpy as np
import random
import itertools
import scipy.stats as stats
import heapq
import matplotlib.pyplot as plt
import math
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import time


def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    d_diags = [float(s.strip()) for s in lines[1: 101]]
    d_offs = [float(s.strip()) for s in lines[102: 202]]
    d_minors = [float(s.strip()) for s in lines[203: 303]]
    d_rows = [float(s.strip()) for s in lines[304: 404]]
    tau_diags = [float(s.strip()) for s in lines[405: 505]]
    tau_offs = [float(s.strip()) for s in lines[506: 606]]
    tau_minors = [float(s.strip()) for s in lines[607: 707]]
    tau_rows = [float(s.strip()) for s in lines[708: 808]]
    """
    plt.scatter(tau_diags, d_diags, edgecolors='r', facecolors='none', label='on-diagonal')
    plt.scatter(tau_rows, d_rows, edgecolors='y', facecolors='none', label='row')
    plt.scatter(tau_minors, d_minors, edgecolors='g', facecolors='none', label='minor')
    plt.scatter(tau_offs, d_offs, edgecolors='b', facecolors='none', label='random')

    t_dmean, d_dmean, t_dstd, d_dstd = np.mean(tau_diags), np.mean(d_diags), np.std(tau_diags), np.std(d_diags)
    t_rmean, d_rmean, t_rstd, d_rstd = np.mean(tau_rows), np.mean(d_rows), np.std(tau_rows), np.std(d_rows)
    t_mmean, d_mmean, t_mstd, d_mstd = np.mean(tau_minors), np.mean(d_minors), np.std(tau_minors), np.std(d_minors)
    t_omean, d_omean, t_ostd, d_ostd = np.mean(tau_offs), np.mean(d_offs), np.std(tau_offs), np.std(d_offs)
    plt.errorbar(t_dmean, d_dmean, xerr=t_dstd, yerr=d_dstd, color='r', ecolor='black',
                 elinewidth=1.5, capsize=3)
    plt.errorbar(t_rmean, d_rmean, xerr=t_rstd, yerr=d_rstd, color='y', ecolor='black',
                 elinewidth=1.5, capsize=3)
    plt.errorbar(t_mmean, d_mmean, xerr=t_mstd, yerr=d_mstd, color='g', ecolor='black',
                 elinewidth=1.5, capsize=3)
    plt.errorbar(t_omean, d_omean, xerr=t_ostd, yerr=d_ostd, color='b', ecolor='black',
                 elinewidth=1.5, capsize=3)

    # plt.title('d vs τ for random minor, off-diagonal, on-diagonal, and row selections')
    plt.ylabel('d', fontsize=10 * 1.5, fontstyle='italic')
    plt.xlabel('τ', fontsize=10 * 1.5, fontstyle='italic')
    plt.legend(loc='upper left', fontsize=10 * 1.5)
    plt.xticks(fontsize=10 * 1.5)
    plt.yticks(fontsize=10 * 1.5)
    plt.show()
    """

    return d_diags, d_offs, d_minors, d_rows, tau_diags, tau_offs, tau_minors, tau_rows


def data_generation(ddiags, doffs, dminors, drows, tdiags, toffs, tminors, trows):
    networks = ['barabasi_albert', 'barabasi_albert', 'bats', 'bats', 'dolphins', 'dolphins',
                'drugusers', 'drugusers', 'elephantseals', 'elephantseals',
                'er_islands','er_islands', 'erdos_renyi', 'erdos_renyi',
                'fitness', 'fitness', 'hall', 'hall', 'highschoolboys','highschoolboys',
                'housefinches', 'housefinches', 'jazz', 'jazz', 'karate', 'karate',
                'LFR', 'LFR', 'lizards', 'lizards', 'nestbox', 'nestbox',
                'netsci', 'netsci', 'pira', 'pira', 'powerlaw', 'powerlaw',
                'surfers', 'surfers', 'tortoises', 'tortoises',
                'voles', 'voles', 'weaverbirds', 'weaverbirds',
                'barabasi_albert', 'barabasi_albert', 'bats', 'bats', 'dolphins', 'dolphins',
                'drugusers', 'drugusers', 'elephantseals', 'elephantseals',
                'er_islands', 'er_islands', 'erdos_renyi', 'erdos_renyi',
                'fitness', 'fitness', 'hall', 'hall', 'highschoolboys', 'highschoolboys',
                'housefinches', 'housefinches', 'jazz', 'jazz', 'karate', 'karate',
                'LFR', 'LFR', 'lizards', 'lizards', 'nestbox', 'nestbox',
                'netsci', 'netsci', 'pira', 'pira', 'powerlaw', 'powerlaw',
                'surfers', 'surfers', 'tortoises', 'tortoises',
                'voles', 'voles', 'weaverbirds', 'weaverbirds',
                ]
    colors = ['diag', 'random', 'minor', 'row']
    bifur_type = ['u', 'D']
    counter = 0
    dependent_var_d, dependent_var_tau, bifurcation_type, homogeneous_type = [], [], [], []
    # dependent_var_d, dependent_var_tau, homogeneous_type = [], [], []
    for _ in range(18400):
        bifurcation_type.append('u')
    for _ in range(18400):
        bifurcation_type.append('D')
    for i in range(46):
        for _ in range(400):
            homogeneous_type.append('homo')
        for _ in range(400):
            homogeneous_type.append('hetero')
    for i in networks:
        for _ in range(400):
            network_type.append(i)
        for j in colors:
            for _ in range(100):
                color_type.append(j)
        dependent_var_d.extend(ddiags[counter: counter + 100])
        dependent_var_d.extend(doffs[counter: counter + 100])
        dependent_var_d.extend(dminors[counter: counter + 100])
        dependent_var_d.extend(drows[counter: counter + 100])
        dependent_var_tau.extend(tdiags[counter: counter + 100])
        dependent_var_tau.extend(toffs[counter: counter + 100])
        dependent_var_tau.extend(tminors[counter: counter + 100])
        dependent_var_tau.extend(trows[counter: counter + 100])

        counter += 100
    return network_type, color_type, dependent_var_d, \
        dependent_var_tau, bifurcation_type, homogeneous_type


if __name__ == '__main__':
    start_time = time.time()
    # d_diags, d_offs, d_minors, d_rows, tau_diags, \
        # tau_offs, tau_minors, tau_rows = read_file('barabasi_albert-d-u-homo-v2data')
    files = ['barabasi_albert-g-u-homo-v2-2026data', 'barabasi_albert-g-u-hetero-v2-2026data',
                 'bats-g-u-homo-v2-2026data', 'bats-g-u-hetero-v2-2026data',
                 'dolphins-g-u-homo-v2-2026data', 'dolphins-g-u-hetero-v2-2026data',
                 'drugusers-g-u-homo-v2-2026data', 'drugusers-g-u-hetero-v2-2026data',
                 'elephantseals-g-u-homo-v2-2026data', 'elephantseals-g-u-hetero-v2-2026data',
                 'er_islands-g-u-homo-v2-2026data', 'er_islands-g-u-hetero-v2-2026data',
                 'erdos_renyi-g-u-homo-v2-2026data', 'erdos_renyi-g-u-hetero-v2-2026data',
                 'fitness-g-u-homo-v2-2026data', 'fitness-g-u-hetero-v2-2026data',
                 'hall-g-u-homo-v2-2026data', 'hall-g-u-hetero-v2-2026data',
                 'highschoolboys-g-u-homo-v2-2026data', 'highschoolboys-g-u-hetero-v2-2026data',
                 'housefinches-g-u-homo-v2-2026data', 'housefinches-g-u-hetero-v2-2026data',
                   'jazz-g-u-homo-v2-2026data', 'jazz-g-u-hetero-v2-2026data',
                   'karate-g-u-homo-v2-2026data', 'karate-g-u-hetero-v2-2026data',
                   'LFR-g-u-homo-v2-2026data', 'LFR-g-u-hetero-v2-2026data',
                    'lizards-g-u-homo-v2-2026data', 'lizards-g-u-hetero-v2-2026data',
                    'nestbox-g-u-homo-v2-2026data', 'nestbox-g-u-hetero-v2-2026data',
                   'netsci-g-u-homo-v2-2026data', 'netsci-g-u-hetero-v2-2026data',
                   'pira-g-u-homo-v2-2026data', 'pira-g-u-hetero-v2-2026data',
                   'powerlaw-g-u-homo-v2-2026data', 'powerlaw-g-u-hetero-v2-2026data',
                    'surfers-g-u-homo-v2-2026data', 'surfers-g-u-hetero-v2-2026data',
                   'tortoises-g-u-homo-v2-2026data', 'tortoises-g-u-hetero-v2-2026data',
                   'voles-g-u-homo-v2-2026data', 'voles-g-u-hetero-v2-2026data',
                   'weaverbirds-g-u-homo-v2-2026data', 'weaverbirds-g-u-hetero-v2-2026data',
             'barabasi_albert-g-D-homo-2026data', 'barabasi_albert-g-D-hetero-2026data',
             'bats-g-D-homo-2026data', 'bats-g-D-hetero-2026data',
             'dolphins-g-D-homo-2026data', 'dolphins-g-D-hetero-2026data',
             'drugusers-g-D-homo-2026data', 'drugusers-g-D-hetero-2026data',
             'elephantseals-g-D-homo-2026data', 'elephantseals-g-D-hetero-2026data',
             'er_islands-g-D-homo-2026data', 'er_islands-g-D-hetero-2026data',
             'erdos_renyi-g-D-homo-2026data', 'erdos_renyi-g-D-hetero-2026data',
             'fitness-g-D-homo-2026data', 'fitness-g-D-hetero-2026data',
             'hall-g-D-homo-2026data', 'hall-g-D-hetero-2026data',
             'highschoolboys-g-D-homo-2026data', 'highschoolboys-g-D-hetero-2026data',
             'housefinches-g-D-homo-2026data', 'housefinches-g-D-hetero-2026data',
             'jazz-g-D-homo-2026data', 'jazz-g-D-hetero-2026data',
             'karate-g-D-homo-2026data', 'karate-g-D-hetero-2026data',
             'LFR-g-D-homo-2026data', 'LFR-g-D-hetero-2026data',
             'lizards-g-D-homo-2026data', 'lizards-g-D-hetero-2026data',
             'nestbox-g-D-homo-2026data', 'nestbox-g-D-hetero-2026data',
             'netsci-g-D-homo-2026data', 'netsci-g-D-hetero-2026data',
             'pira-g-D-homo-2026data', 'pira-g-D-hetero-2026data',
             'powerlaw-g-D-homo-2026data', 'powerlaw-g-D-hetero-2026data',
             'surfers-g-D-homo-2026data', 'surfers-g-D-hetero-2026data',
             'tortoises-g-D-homo-2026data', 'tortoises-g-D-hetero-2026data',
             'voles-g-D-homo-2026data', 'voles-g-D-hetero-2026data',
             'weaverbirds-g-D-homo-2026data', 'weaverbirds-g-D-hetero-2026data'
             ]
    d_tol_diags, d_tol_offs, d_tol_minors, d_tol_rows = [], [], [], []
    tau_tol_diags, tau_tol_offs, tau_tol_minors, tau_tol_rows = [], [], [], []
    for i in files:
        d_diags, d_offs, d_minors, d_rows, tau_diags, tau_offs, tau_minors, tau_rows = read_file(i)
        tau_minors = [-1 * x for x in tau_minors]
        tau_offs = [-1 * x for x in tau_offs]
        tau_diags = [-1 * x for x in tau_diags]
        tau_rows = [-1 * x for x in tau_rows]
        d_tol_diags.extend(d_diags)
        d_tol_offs.extend(d_offs)
        d_tol_minors.extend(d_minors)
        d_tol_rows.extend(d_rows)
        tau_tol_diags.extend(tau_diags)
        tau_tol_offs.extend(tau_offs)
        tau_tol_minors.extend(tau_minors)
        tau_tol_rows.extend(tau_rows)
    network_type, color_type, dependent_var_d, dependent_var_tau, bifurcation_type\
        = [], [], [], [], []
    network_type, color_type, dependent_var_d, dependent_var_tau, bifurcation_type, homogeneous_type = \
        data_generation(d_tol_diags, d_tol_offs, d_tol_minors, d_tol_rows, tau_tol_diags,
                        tau_tol_offs, tau_tol_minors, tau_tol_rows)
    print(len(network_type), len(color_type), len(dependent_var_d), len(dependent_var_tau))
    data = {
        'network_types': network_type,
        'color_types': color_type,
        'bifurcation_types': bifurcation_type,
        'homogeneous_types': homogeneous_type,
        'dependent_vars': dependent_var_tau
    }

    df = pd.DataFrame(data)

    model = ols('dependent_vars ~ C(network_types)  + C(bifurcation_types) + '
                'C(homogeneous_types, Treatment(reference="homo")) + '
                'C(color_types, Treatment(reference="diag"))', data=df).fit()
    # model = ols('dependent_vars ~ C(network_types)', data=df).fit()
    coefficients = model.params
    p_values = model.pvalues
    formatted_results = pd.DataFrame({
        "Coefficient": [f"{coef:.10f}" for coef in coefficients],
        "P-Value": [f"{pval:.10f}" for pval in p_values]
    })
    print(formatted_results)
    anova_results = anova_lm(model)
    pd.options.display.float_format = "{:.10f}".format
    # print(anova_results)
    # Print model summary, which includes coefficients relative to the baseline
    # print(model.summary())
    with open('2026 gene model summary tau.txt', 'w') as f:
        f.write(model.summary().as_text())

    std_dep_var = np.std(df['dependent_vars'], ddof=1)
    coefficients = model.params
    # print('coefficients', type(coefficients))
    effect_sizes = coefficients / std_dep_var
    baseline_mean = df.loc[df['color_types'] == "diag", 'dependent_vars'].mean()
    # print('baseline effect size is ', baseline_mean / std_dep_var)
    # print("Effect Sizes:\n", effect_sizes)
    result_str = "\nEffect Sizes:\n" + effect_sizes.to_string()
    with open('2026 gene model ANOVA tau.txt', 'w') as f:
        f.write("ANOVA Results:\n")
        f.write(anova_results.to_string())
        f.write("\n\nStandard deviation of dependent variable is " + str(std_dep_var) + "\n")
        f.write(result_str)
        f.write("\nBaseline effect size is:\n")
        f.write(str(baseline_mean/std_dep_var))
    anova_results.to_csv("2026 anova_results tau gene.csv")
    # if (anova_results['PR(>F)'][1] == 0):
    # print (123456789)
    # print(anova_results)
    # anova_results.to_csv("anova_results22.csv", index=True)  # Save the table
    tukey_result = pairwise_tukeyhsd(endog=df['dependent_vars'], groups=df['color_types'], alpha=0.05)
    tukey_summary = pd.DataFrame(
        data=tukey_result.summary().data[1:],  # Skip header row
        columns=tukey_result.summary().data[0]  # Use header row
    )
    tukey_summary["p-adj"] = tukey_summary["p-adj"].astype(float).map("{:.10f}".format)
    print(tukey_summary)
    result_str = tukey_result.summary().as_text()
    with open('2026 gene model tukey tau.txt', 'w') as f:
        f.write(tukey_summary.to_string(index=False))
    # print(tukey_result)

    data = {
        'network_types': network_type,
        'color_types': color_type,
        'bifurcation_types': bifurcation_type,
        'homogeneous_types': homogeneous_type,
        'dependent_vars': dependent_var_d
    }

    df = pd.DataFrame(data)

    model = ols('dependent_vars ~ C(network_types)  + C(bifurcation_types) + '
                'C(homogeneous_types, Treatment(reference="homo")) + '
                'C(color_types, Treatment(reference="diag"))', data=df).fit()
    # model = ols('dependent_vars ~ C(network_types)', data=df).fit()
    coefficients = model.params
    p_values = model.pvalues
    formatted_results = pd.DataFrame({
        "Coefficient": [f"{coef:.10f}" for coef in coefficients],
        "P-Value": [f"{pval:.10f}" for pval in p_values]
    })
    print(formatted_results)
    anova_results = anova_lm(model)
    pd.options.display.float_format = "{:.10f}".format
    # print(anova_results)
    # Print model summary, which includes coefficients relative to the baseline
    # print(model.summary())
    with open('2026 gene model summary d.txt', 'w') as f:
        f.write(model.summary().as_text())

    std_dep_var = np.std(df['dependent_vars'], ddof=1)
    coefficients = model.params
    # print('coefficients', type(coefficients))
    effect_sizes = coefficients / std_dep_var
    baseline_mean = df.loc[df['color_types'] == "diag", 'dependent_vars'].mean()
    # print('baseline effect size is ', baseline_mean / std_dep_var)
    # print("Effect Sizes:\n", effect_sizes)
    result_str = "\nEffect Sizes:\n" + effect_sizes.to_string()
    with open('2026 gene model ANOVA d.txt', 'w') as f:
        f.write("ANOVA Results:\n")
        f.write(anova_results.to_string())
        f.write("\n\nStandard deviation of dependent variable is " + str(std_dep_var) + "\n")
        f.write(result_str)
        f.write("\nBaseline effect size is:\n")
        f.write(str(baseline_mean / std_dep_var))
    # with open('anova_results23.csv', 'w') as f:
        # f.write(anova_results)
    anova_results.to_csv("2026 anova_results d gene.csv")

    # if (anova_results['PR(>F)'][1] == 0):
    # print (123456789)
    # print(anova_results)
    # anova_results.to_csv("anova_results22.csv", index=True)  # Save the table
    tukey_result = pairwise_tukeyhsd(endog=df['dependent_vars'], groups=df['color_types'], alpha=0.05)
    tukey_summary = pd.DataFrame(
        data=tukey_result.summary().data[1:],  # Skip header row
        columns=tukey_result.summary().data[0]  # Use header row
    )
    tukey_summary["p-adj"] = tukey_summary["p-adj"].astype(float).map("{:.10f}".format)
    print(tukey_summary)
    result_str = tukey_result.summary().as_text()
    with open('2026 gene model tukey d.txt', 'w') as f:
        f.write(tukey_summary.to_string(index=False))

    end_time = time.time()
    print(end_time - start_time)