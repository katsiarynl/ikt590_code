import numpy as np 
from scipy.stats import mannwhitneyu
from itertools import combinations
from statsmodels.stats.multitest import multipletests

def compute_clause_length_statistics(data_pos, data_neg, dataset_name='BreastMNIST', target_class=0):
    specialists = list(data_pos.keys())

    print(f"\n{'='*90}")
    print(f"Clause Length Statistics | {dataset_name} | Class {target_class}")
    print(f"{'='*90}")
    print(f"  {'Specialist':<25} {'Polarity':<12} {'n':>6} {'Mean':>10} {'Median':>10} {'Std':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")

    results = {}
    for s in specialists:
        pos = np.array(data_pos[s])
        neg = np.array(data_neg[s])

        stats_pos = {
            'n': len(pos),
            'mean': np.mean(pos),
            'median': np.median(pos),
            'std': np.std(pos, ddof=1)
        }
        stats_neg = {
            'n': len(neg),
            'mean': np.mean(neg),
            'median': np.median(neg),
            'std': np.std(neg, ddof=1)
        }

        results[s] = {'pos': stats_pos, 'neg': stats_neg}

        print(f"  {s:<25} {'Positive':<12} {stats_pos['n']:>6} {stats_pos['mean']:>10.2f} {stats_pos['median']:>10.2f} {stats_pos['std']:>10.2f}")
        print(f"  {'':<25} {'Negative':<12} {stats_neg['n']:>6} {stats_neg['mean']:>10.2f} {stats_neg['median']:>10.2f} {stats_neg['std']:>10.2f}")
        print(f"  {'-'*25} {'-'*12} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")

    print(f"{'='*90}\n")
    return results  


from scipy.stats import mannwhitneyu

def format_p(p):
    if p < 0.001:
        return f"{p:.2e}"
    else:
        return f"{p:.3f}"

def run_mannwhitney_pos_neg_length(data_pos, data_neg, dataset_name='BreastMNIST', target_class=0, alpha=0.05):
    specialists = list(data_pos.keys())
    print(f"\n{'='*70}")
    print(f"Mann-Whitney U — Positive vs Negative Clause Lengths")
    print(f"Dataset: {dataset_name} | Class {target_class}")
    print(f"{'='*70}")
    print("H0: Positive and negative clause length distributions are identical.")
    print("H1: Positive and negative clause length distributions differ.")
    print()
    results = []
    print(f"  {'Specialist':<25} {'n_pos':>6} {'n_neg':>6} {'U-stat':>10} {'p-value':>12} {'Effect (r)':>12} {'Reject H0':>10}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")
    for s in specialists:
        stat, p = mannwhitneyu(data_pos[s], data_neg[s], alternative='two-sided')
        n1, n2 = len(data_pos[s]), len(data_neg[s])
        r = 1 - (2 * stat) / (n1 * n2)
        sig = '✓' if p < alpha else '✗'
        results.append((s, stat, p, r, n1, n2))
        print(f"  {s:<25} {n1:>6} {n2:>6} {stat:>10.1f} {format_p(p):>12} {r:>12.4f} {sig:>10}")
    print(f"\n  Significance level: α = {alpha}")
    print(f"{'='*70}\n")
    return results



from scipy.stats import kruskal

def run_kruskalwallis_across_specialists(data_pos, data_neg, dataset_name='BreastMNIST', target_class=0, alpha=0.05):
    specialists = list(data_pos.keys())

    print(f"\n{'='*70}")
    print(f"Kruskal-Wallis — Clause Lengths Across Specialists")
    print(f"Dataset: {dataset_name} | Class {target_class}")
    print(f"{'='*70}")
    print(f"H0: Clause length distributions are identical across all specialists.")
    print(f"H1: At least one specialist has a different clause length distribution.")
    print()

    print(f"  {'Polarity':<12} {'H-statistic':>12} {'p-value':>12} {'Reject H0':>10}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    for polarity, data in [('Positive', data_pos), ('Negative', data_neg)]:
        stat, p = kruskal(*[data[s] for s in specialists])
        sig = '✓' if p < alpha else '✗'
        print(f"  {polarity:<12} {stat:>12.4f} {p:>12.4f} {sig:>10}")

    print(f"\n  Significance level: α = {alpha}")
    print(f"{'='*70}\n")

def run_posthoc_mannwhitney(data_pos, data_neg, dataset_name='BreastMNIST', target_class=0, alpha=0.05):
    specialists = list(data_pos.keys())
    pairs = list(combinations(specialists, 2))
    print(f"\n{'='*80}")
    print(f"Post-hoc Pairwise Mann-Whitney U — Across Specialists")
    print(f"Dataset: {dataset_name} | Class {target_class}")
    print(f"{'='*80}")
    print(f"H0: Clause length distributions are identical for a given pair of specialists.")
    print(f"H1: Clause length distributions differ for a given pair of specialists.")
    for polarity, data in [('Positive', data_pos), ('Negative', data_neg)]:
        print(f"\n--- {polarity} clauses ---")
        print(f"  {'Pair':<45} {'U-stat':>10} {'p-value':>14} {'Effect (r)':>12} {'Reject H0':>10}")
        print(f"  {'-'*45} {'-'*10} {'-'*14} {'-'*12} {'-'*10}")
        for s1, s2 in pairs:
            stat, p = mannwhitneyu(data[s1], data[s2], alternative='two-sided')
            n1, n2 = len(data[s1]), len(data[s2])
            r = 1 - (2 * stat) / (n1 * n2)
            sig = '✓' if p < alpha else '✗'
            pair_str = f"{s1} vs {s2}"
            print(f"  {pair_str:<45} {stat:>10.1f} {format_p(p):>14} {r:>12.4f} {sig:>10}")
    print(f"\n  Significance level: α = {alpha}")
    print(f"{'='*80}\n")