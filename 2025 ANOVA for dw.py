import numpy as np
import pandas as pd
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
                'barabasi_albert', 'barabasi_albert', 'bats', 'bats', 'dolphins', 'dolphins',
                'drugusers', 'drugusers', 'elephantseals', 'elephantseals',
                'er_islands', 'er_islands', 'erdos_renyi', 'erdos_renyi',
                'fitness', 'fitness', 'hall', 'hall', 'highschoolboys', 'highschoolboys',
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
    dependent_var_d, dependent_var_tau, bifurcation_type, homogeneous_type, \
        updown_type = [], [], [], [], []
    # dependent_var_d, dependent_var_tau, homogeneous_type = [], [], []
    for _ in range(36800):
        updown_type.append('up')
    for _ in range(36800):
        updown_type.append('down')
    for _ in range(18400):
        bifurcation_type.append('u')
    for _ in range(18400):
        bifurcation_type.append('D')
    for _ in range(18400):
        bifurcation_type.append('u')
    for _ in range(18400):
        bifurcation_type.append('D')
    for i in range(46):
        for _ in range(400):
            homogeneous_type.append('homo')
        for _ in range(400):
            homogeneous_type.append('hetero')
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
        dependent_var_tau, bifurcation_type, homogeneous_type, updown_type


if __name__ == '__main__':
    start_time = time.time()
    # d_diags, d_offs, d_minors, d_rows, tau_diags, \
        # tau_offs, tau_minors, tau_rows = read_file('barabasi_albert-d-u-homo-v2data')
    files = [
        'barabasi_albert-d-u-homo-2026data', 'barabasi_albert-d-u-hetero-2026data',
        'bats-d-u-homo-2026data', 'bats-d-u-hetero-2026data',
        'dolphins-d-u-homo-2026data', 'dolphins-d-u-hetero-2026data',
        'drugusers-d-u-homo-2026data', 'drugusers-d-u-hetero-2026data',
        'elephantseals-d-u-homo-2026data', 'elephantseals-d-u-hetero-2026data',
        'er_islands-d-u-homo-2026data', 'er_islands-d-u-hetero-2026data',
        'erdos_renyi-d-u-homo-2026data', 'erdos_renyi-d-u-hetero-2026data',
        'fitness-d-u-homo-2026data', 'fitness-d-u-hetero-2026data',
        'hall-d-u-homo-2026data', 'hall-d-u-hetero-2026data',
        'highschoolboys-d-u-homo-2026data', 'highschoolboys-d-u-hetero-2026data',
        'housefinches-d-u-homo-2026data', 'housefinches-d-u-hetero-2026data',
        'jazz-d-u-homo-2026data', 'jazz-d-u-hetero-2026data',
        'karate-d-u-homo-2026data', 'karate-d-u-hetero-2026data',
        'LFR-d-u-homo-2026data', 'LFR-d-u-hetero-2026data',
        'lizards-d-u-homo-2026data', 'lizards-d-u-hetero-2026data',
        'nestbox-d-u-homo-2026data', 'nestbox-d-u-hetero-2026data',
        'netsci-d-u-homo-2026data', 'netsci-d-u-hetero-2026data',
        'pira-d-u-homo-2026data', 'pira-d-u-hetero-2026data',
        'powerlaw-d-u-homo-2026data', 'powerlaw-d-u-hetero-2026data',
        'surfers-d-u-homo-2026data', 'surfers-d-u-hetero-2026data',
        'tortoises-d-u-homo-2026data', 'tortoises-d-u-hetero-2026data',
        'voles-d-u-homo-2026data', 'voles-d-u-hetero-2026data',
        'weaverbirds-d-u-homo-2026data', 'weaverbirds-d-u-hetero-2026data',
        'barabasi_albert-d-D-homo-2026data', 'barabasi_albert-d-D-hetero-2026data',
        'bats-d-D-homo-2026data', 'bats-d-D-hetero-2026data',
        'dolphins-d-D-homo-2026data', 'dolphins-d-D-hetero-2026data',
        'drugusers-d-D-homo-2026data', 'drugusers-d-D-hetero-2026data',
        'elephantseals-d-D-homo-2026data', 'elephantseals-d-D-hetero-2026data',
        'er_islands-d-D-homo-2026data', 'er_islands-d-D-hetero-2026data',
        'erdos_renyi-d-D-homo-2026data', 'erdos_renyi-d-D-hetero-2026data',
        'fitness-d-D-homo-2026data', 'fitness-d-D-hetero-2026data',
        'hall-d-D-homo-2026data', 'hall-d-D-hetero-2026data',
        'highschoolboys-d-D-homo-2026data', 'highschoolboys-d-D-hetero-2026data',
        'housefinches-d-D-homo-2026data', 'housefinches-d-D-hetero-2026data',
        'jazz-d-D-homo-2026data', 'jazz-d-D-hetero-2026data',
        'karate-d-D-homo-2026data', 'karate-d-D-hetero-2026data',
        'LFR-d-D-homo-2026data', 'LFR-d-D-hetero-2026data',
        'lizards-d-D-homo-2026data', 'lizards-d-D-hetero-2026data',
        'nestbox-d-D-homo-2026data', 'nestbox-d-D-hetero-2026data',
        'netsci-d-D-homo-2026data', 'netsci-d-D-hetero-2026data',
        'pira-d-D-homo-2026data', 'pira-d-D-hetero-2026data',
        'powerlaw-d-D-homo-2026data', 'powerlaw-d-D-hetero-2026data',
        'surfers-d-D-homo-2026data', 'surfers-d-D-hetero-2026data',
        'tortoises-d-D-homo-2026data', 'tortoises-d-D-hetero-2026data',
        'voles-d-D-homo-2026data', 'voles-d-D-hetero-2026data',
        'weaverbirds-d-D-homo-2026data', 'weaverbirds-d-D-hetero-2026data',
        'barabasi_albert-d-u-homo-down-2026data', 'barabasi_albert-d-u-hetero-down-2026data',
        'bats-d-u-homo-down-2026data', 'bats-d-u-hetero-down-2026data',
        'dolphins-d-u-homo-down-2026data', 'dolphins-d-u-hetero-down-2026data',
        'drugusers-d-u-homo-down-2026data', 'drugusers-d-u-hetero-down-2026data',
        'elephantseals-d-u-homo-down-2026data', 'elephantseals-d-u-hetero-down-2026data',
        'er_islands-d-u-homo-down-2026data', 'er_islands-d-u-hetero-down-2026data',
        'erdos_renyi-d-u-homo-down-2026data', 'erdos_renyi-d-u-hetero-down-2026data',
        'fitness-d-u-homo-down-2026data', 'fitness-d-u-hetero-down-2026data',
        'hall-d-u-homo-down-2026data', 'hall-d-u-hetero-down-2026data',
        'highschoolboys-d-u-homo-down-2026data', 'highschoolboys-d-u-hetero-down-2026data',
        'housefinches-d-u-homo-down-2026data', 'housefinches-d-u-hetero-down-2026data',
        'jazz-d-u-homo-down-2026data', 'jazz-d-u-hetero-down-2026data',
        'karate-d-u-homo-down-2026data', 'karate-d-u-hetero-down-2026data',
        'LFR-d-u-homo-down-2026data', 'LFR-d-u-hetero-down-2026data',
        'lizards-d-u-homo-down-2026data', 'lizards-d-u-hetero-down-2026data',
        'nestbox-d-u-homo-down-2026data', 'nestbox-d-u-hetero-down-2026data',
        'netsci-d-u-homo-down-2026data', 'netsci-d-u-hetero-down-2026data',
        'pira-d-u-homo-down-2026data', 'pira-d-u-hetero-down-2026data',
        'powerlaw-d-u-homo-down-2026data', 'powerlaw-d-u-hetero-down-2026data',
        'surfers-d-u-homo-down-2026data', 'surfers-d-u-hetero-down-2026data',
        'tortoises-d-u-homo-down-2026data', 'tortoises-d-u-hetero-down-2026data',
        'voles-d-u-homo-down-2026data', 'voles-d-u-hetero-down-2026data',
        'weaverbirds-d-u-homo-down-2026data', 'weaverbirds-d-u-hetero-down-2026data',
        'barabasi_albert-d-D-homo-down-2026data', 'barabasi_albert-d-D-hetero-down-2026data',
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
        'weaverbirds-d-D-homo-down-2026data', 'weaverbirds-d-D-hetero-down-2026data'
    ]
    counter = 0
    d_tol_diags, d_tol_offs, d_tol_minors, d_tol_rows = [], [], [], []
    tau_tol_diags, tau_tol_offs, tau_tol_minors, tau_tol_rows = [], [], [], []
    for i in files:
        d_diags, d_offs, d_minors, d_rows, tau_diags, tau_offs, tau_minors, tau_rows = read_file(i)
        if counter >= 92:
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
        counter += 1
    network_type, color_type, dependent_var_d, dependent_var_tau, bifurcation_type, updown_type\
        = [], [], [], [], [], []
    network_type, color_type, dependent_var_d, dependent_var_tau, bifurcation_type, \
        homogeneous_type, updown_type = \
        data_generation(d_tol_diags, d_tol_offs, d_tol_minors, d_tol_rows, tau_tol_diags,
                        tau_tol_offs, tau_tol_minors, tau_tol_rows)
    print(len(network_type), len(color_type), len(dependent_var_d), len(dependent_var_tau))
    data = {
        'network_types': network_type,
        'color_types': color_type,
        'bifurcation_types': bifurcation_type,
        'homogeneous_types': homogeneous_type,
        'dependent_vars': dependent_var_tau,
        'updown_types': updown_type
    }

    df = pd.DataFrame(data)

    model = ols('dependent_vars ~ C(network_types)  + C(bifurcation_types) + '
                'C(homogeneous_types, Treatment(reference="homo")) + '
                'C(updown_types, Treatment(reference="up")) + '
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
    with open('2026 dw model summary tau.txt', 'w') as f:
        f.write(model.summary().as_text())

    std_dep_var = np.std(df['dependent_vars'], ddof=1)
    coefficients = model.params
    # print('coefficients', type(coefficients))
    effect_sizes = coefficients / std_dep_var
    baseline_mean = df.loc[df['color_types'] == "diag", 'dependent_vars'].mean()
    # print('baseline effect size is ', baseline_mean / std_dep_var)
    # print("Effect Sizes:\n", effect_sizes)
    result_str = "\nEffect Sizes:\n" + effect_sizes.to_string()
    with open('2026 dw model ANOVA tau.txt', 'w') as f:
        f.write("ANOVA Results:\n")
        f.write(anova_results.to_string())
        f.write("\n\nStandard deviation of dependent variable is " + str(std_dep_var) + "\n")
        f.write(result_str)
        f.write("\nBaseline effect size is:\n")
        f.write(str(baseline_mean/std_dep_var))
    anova_results.to_csv("2026 anova_results tau dw.csv")

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
    result_str = tukey_result.summary().as_text()
    with open('2026 dw model tukey tau.txt', 'w') as f:
        f.write(tukey_summary.to_string(index=False))
    # print(tukey_result)

    data = {
        'network_types': network_type,
        'color_types': color_type,
        'bifurcation_types': bifurcation_type,
        'homogeneous_types': homogeneous_type,
        'dependent_vars': dependent_var_d,
        'updown_types': updown_type
    }

    df = pd.DataFrame(data)

    model = ols('dependent_vars ~ C(network_types)  + C(bifurcation_types) + '
                'C(homogeneous_types, Treatment(reference="homo")) + '
                'C(updown_types, Treatment(reference="up")) + '
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
    with open('2026 dw model summary d.txt', 'w') as f:
        f.write(model.summary().as_text())

    std_dep_var = np.std(df['dependent_vars'], ddof=1)
    coefficients = model.params
    # print('coefficients', type(coefficients))
    effect_sizes = coefficients / std_dep_var
    baseline_mean = df.loc[df['color_types'] == "diag", 'dependent_vars'].mean()
    # print('baseline effect size is ', baseline_mean / std_dep_var)
    # print("Effect Sizes:\n", effect_sizes)
    result_str = "\nEffect Sizes:\n" + effect_sizes.to_string()
    with open('2026 dw model ANOVA d.txt', 'w') as f:
        f.write("ANOVA Results:\n")
        f.write(anova_results.to_string())
        f.write("\n\nStandard deviation of dependent variable is " + str(std_dep_var) + "\n")
        f.write(result_str)
        f.write("\nBaseline effect size is:\n")
        f.write(str(baseline_mean / std_dep_var))
    anova_results.to_csv("2026 anova_results d dw.csv")

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
    result_str = tukey_result.summary().as_text()
    with open('2026 dw model tukey d.txt', 'w') as f:
        f.write(tukey_summary.to_string(index=False))
    # print(tukey_result)

    end_time = time.time()
    print(end_time - start_time)