import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from scipy.stats import ttest_rel

n_groups = 3
n_sub_per_group = 10
sub_ids = np.array(range(n_sub_per_group) * n_groups)
amp = np.array([1] * n_sub_per_group + [2] * n_sub_per_group + [3] * n_sub_per_group)
current_thresh = np.concatenate((
                            np.random.randn(n_sub_per_group) + 0,
                            np.random.randn(n_sub_per_group) + 3,
                            np.random.randn(n_sub_per_group) + 0
                            ))

df = pd.DataFrame({'sub_ids': sub_ids, 'amp': amp, 'current_tresh': current_thresh})
#print df

# repeated measured ANOVA
aovrm = AnovaRM(df, 'current_tresh', 'sub_ids', within=['amp'])
res = aovrm.fit()
print(res)

# post-hoc: paired-t-test with Bonferroni correction
group1 = current_thresh[:n_sub_per_group]
group2 = current_thresh[n_sub_per_group:2*n_sub_per_group]
group3 = current_thresh[2*n_sub_per_group:]

_, p12 = ttest_rel(group1, group2)
_, p23 = ttest_rel(group2, group3)
_, p31 = ttest_rel(group3, group1)

print 'P 1-2: ', p12*3.
print 'P 2-3: ', p23*3.
print 'P 3-1: ', p31*3.


def compute_repeated_ANOVA_and_posthoc_paired_test_with_Bonferroni(data):
    """
    Subjects measured in each group should be the same.
    :param data: rows: subjects, columns: different measurements
    :return:
    """
    n_subjects = np.shape(data)[0]
    n_groups = np.shape(data)[1]
    subject_ids = np.arange(n_subjects, dtype=int)

    # rearrange data for the repeated measures ANOVA
    group_indicator = np.array([0] * n_subjects + [1] * n_subjects + [2] * n_subjects)
    data_concatenated = np.concatenate(data.T)

    df = pd.DataFrame({'subject_ids': subject_ids, 'group_indicator': group_indicator,
                       'data_concatenated': data_concatenated})
    print df

    # repeated measured ANOVA
    aovrm = AnovaRM(df, 'data_concatenated', 'subject_ids', within=['group_indicator'])
    res = aovrm.fit()
    print(res)

    # post-hoc: paired-t-test with Bonferroni correction
    _, p12 = ttest_rel(data[:, 0], data[:, 1])
    _, p23 = ttest_rel(data[:, 1], data[:, 2])
    _, p31 = ttest_rel(data[:, 2], data[:, 0])

    print 'P 1-2: ', p12 * 3.
    print 'P 2-3: ', p23 * 3.
    print 'P 3-1: ', p31 * 3.