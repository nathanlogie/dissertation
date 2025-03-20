import pandas as pd
import scipy.stats as stats
from itertools import combinations
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def analyze_results(filename):
    # Load dataset
    df = pd.read_csv(filename)

    # Group by dataset and model
    results = {}
    for (dataset, model), group in df.groupby(["Dataset", "Model"]):
        results.setdefault(dataset, {}).setdefault(model, {})

        strategies = group["Batch Selection Strategy"].unique()

        # Analyze each metric separately
        for metric in ["Accuracy", "Bal. Acc.", "Prec.", "Rec.", "F1 Score",
                       "Disp. Impact", "Stat. Parity Diff.", "Avg. Odds Diff.", "Eq. Opp. Diff."]:
            metric_values = {s: group.loc[group["Batch Selection Strategy"] == s, metric].values for s in strategies}

            # Check normality
            normal = all(stats.shapiro(values).pvalue > 0.05 for values in metric_values.values())

            # Select test
            if normal:
                # One-way ANOVA for normally distributed data
                f_stat, p_value = stats.f_oneway(*metric_values.values())
                test_used = "ANOVA"
            else:
                # Friedman test for non-normal distributions
                f_stat, p_value = stats.friedmanchisquare(*metric_values.values())
                test_used = "Friedman"

            if p_value < 0.1:  # Significant differences found
                # Post-hoc tests to determine which strategies differ
                pairs = list(combinations(strategies, 2))
                significant_pairs = []

                for s1, s2 in pairs:
                    if normal:
                        # Tukey HSD for ANOVA
                        tukey = pairwise_tukeyhsd(group[metric], group["Batch Selection Strategy"])
                        for res in tukey.summary().data[1:]:
                            if res[0] == s1 and res[1] == s2 and res[4] < 0.1:
                                significant_pairs.append((s1, s2))
                    else:
                        # Wilcoxon Signed-Rank Test for non-parametric data
                        stat, p = stats.wilcoxon(metric_values[s1], metric_values[s2])
                        if p < 0.1:
                            significant_pairs.append((s1, s2))

                results[dataset][model][metric] = {"Test": test_used, "Significant Pairs": significant_pairs}

    # Print results in requested format
    for dataset, models in results.items():
        print(f"Dataset: {dataset}")
        for model, metrics in models.items():
            print(f"  Model: {model}")
            for metric, details in metrics.items():
                print(f"    {metric} ({details['Test']}): {details['Significant Pairs']}")
            print()


# Run the analysis
analyze_results("batching_results.csv")
