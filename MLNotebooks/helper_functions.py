import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
from matplotlib.patches import Rectangle
from IPython.display import display
from scipy.stats import hmean
from sklearn.metrics import precision_recall_curve, confusion_matrix, accuracy_score, roc_auc_score, f1_score, \
    precision_score, recall_score, average_precision_score
import matplotlib.pyplot as plt


def load_and_filter_dataset(path_dfvz='../DataFiles/experimental_data_set.csv',
                            path_df20='../DataFiles/df20.csv',
                            apply_filter='CPXXX'):
    """
    Loads dfvz and df20 dataframes based on provided paths. Can also apply filter to the data as needed.
    Args:
        path_dfvz: filepath to dfvz (default '../DataFiles/experiment_data_set.csv')
        path_df20: filepath to df20 (default '../DataFiles/df20.csv')
        apply_filter: Indicates which type of filter to apply None, 'CPXXX', 'PXXX','C' (default 'CPXXX')

    Returns:
        Returns dfvz, df20
    """

    dfvz = pd.read_csv(path_dfvz, engine='c', index_col=0)
    df20 = pd.read_csv(path_df20, engine='c', index_col=0)
    len_before_filter = np.array([len(dfvz), len(df20)])

    if apply_filter is not None:
        idx_filtered = filter_data_idx(dfvz, apply_filter)
        dfvz = dfvz.loc[idx_filtered]
        df20 = df20.loc[idx_filtered]
        len_after_filter = np.array([len(dfvz), len(df20)])
        print(f'Len Before Filtering for DFVZ and DF20: {len_before_filter}\nLen After Filter: {len_after_filter}')

    return dfvz, df20


def filter_data_idx(df, filter_seqs='CPXXX'):
    """
    Take in a data frame and returns an index of items that should remain once the filter is applied.
    Args:
        df: dataframe to use for filtering. Must contain a column name 'seq' that has sequences to filter from.
        filter_seqs: String. Must be one of 'CPXXX', 'PXXX', 'C' indicating the type of filter to apply (default 'CPXXX')

    Returns:
        Returns none if an incorrect filter_seqs argument is provided.
        Otherwise, returns the index as a list of the sequences that should remain in
        the df after applying the filter
    """
    # Remove all sequences with C, PXXX but not PPXX
    # Find all sequences containing cysteines:
    filter_cysteine = df['seq'].str.contains('C')
    # Find all sequences containing PXXX but not PPXX
    filter_pxxx = df['seq'].str.contains('(^P[^P])')
    # Finding all sequences that either contain a C or PPXX or both
    filter_c_pxxx = df['seq'].str.contains('(^P[^P])|([C])')

    print(f"""
    DF Contains: 
    Sequences Containing C: {np.count_nonzero(filter_cysteine)}
    Sequences Containing PXXX (Not PPXX): {np.count_nonzero(filter_pxxx)}
    Combine the two above filters with or: {np.count_nonzero(filter_c_pxxx)} 
    """)

    if filter_seqs.upper() == 'C':
        idx_inverted = df[~filter_cysteine].index.to_list()
    elif filter_seqs.upper() == 'PXXX':
        idx_inverted = df[~filter_pxxx].index.to_list()
    elif filter_seqs.upper() == 'CPXXX':
        idx_inverted = df[~filter_c_pxxx].index.to_list()
    else:
        print('Invalid Filter Entry. Please select one of these options: "C", "PXXX", "CPXXX"')
        return None

    return idx_inverted


def pre_process_data(dfvz, df20, threshold=0.05):
    """
    Scale and pre-process data and store it in a dictionary that can be easily accessed!

    Args:
        dfvz: dfvz dataframe containing chemical features
        df20: df20 dataframe containing sequence patterns
        threshold: threshold to use to mark high and low classes. I.e what percentile of the population will be
        labelled 'HIGH'. Same threshold is used for 'LOW'. Be default this is set to top 5 percentile and bottom
        5 percentile (0.05)

    Returns:
        X_dict: dictionary of features (both unscaled and scaled)
        Xname_dict: dictionary of feature keys and corresponding column names from dataframes
        y_dict: dictionary containing labels encoded in 4 different formats ('real', '3class', 'low', 'high')
    """

    standard_scaler = preprocessing.StandardScaler()

    Xp = df20.loc[:, '..AA':'V...'].values  # sequence pattern descriptors #(67278, 2396)
    Xp_names = df20.loc[:, '..AA':'V...'].columns
    standard_scaler.fit(Xp)
    Xp_s = standard_scaler.transform(Xp)

    Xz = dfvz.loc[:, 'z1.1':'z4.3'].values  # zscale descriptors #(67278, 12)
    Xz_names = dfvz.loc[:, 'z1.1':'z4.3'].columns
    standard_scaler.fit(Xz)
    Xz_s = standard_scaler.transform(Xz)

    Xv = dfvz.loc[:, 'vhse1.1':'vhse4.8'].values  # vhse descriptors #(67278, 32)
    Xv_names = dfvz.loc[:, 'vhse1.1':'vhse4.8'].columns
    standard_scaler.fit(Xv)
    Xv_s = standard_scaler.transform(Xv)

    Xvz_names = list(Xz_names) + list(Xv_names)  # zscale and vhse combined #(67278, 44)
    Xvz = dfvz.loc[:, Xvz_names].values
    standard_scaler.fit(Xvz)
    Xvz_s = standard_scaler.transform(Xvz)

    Xpvz_names = list(Xvz_names) + list(Xp_names)  # pattern and zscale, vhse combined #(67278, 2440)
    Xpvz = pd.concat([dfvz.loc[:, Xvz_names], df20.loc[:, list(Xp_names)]], axis=1).values
    standard_scaler.fit(Xpvz)
    Xpvz_s = standard_scaler.transform(Xpvz)

    y = dfvz['log.label'].values.reshape(-1, 1)

    # The following dictionary makes it much easier to main and access data.
    keys = ['Xp', 'Xp_s', 'Xz', 'Xz_s', 'Xv', 'Xv_s', 'Xvz', 'Xvz_s', 'Xpvz', 'Xpvz_s']
    vals = [Xp, Xp_s, Xz, Xz_s, Xv, Xv_s, Xvz, Xvz_s, Xpvz, Xpvz_s]
    name_vals = [Xp_names, Xp_names, Xz_names, Xz_names, Xv_names, Xv_names, Xvz_names, Xvz_names, Xpvz_names,
                 Xpvz_names]

    X_dict = dict(zip(keys, vals))
    Xname_dict = dict(zip(keys, name_vals))

    # Y - Values, subdivided according to this:
    # Real: Real Values used for regressor
    # 3class: 0, 1, 2 - Bottom 5%, Middle 90%, Top 5%
    threshold = threshold
    y_three_class = pd.qcut(dfvz['log.label'], q=[0, threshold, 1 - threshold, 1], labels=False).values
    y_low = np.array([True if x == 0 else False for x in y_three_class])
    y_high = np.array([True if x == 2 else False for x in y_three_class])
    ykey = ['real', '3class', 'low', 'high']
    yval = [y, y_three_class, y_low, y_high]
    y_dict = dict(zip(ykey, yval))

    return X_dict, Xname_dict, y_dict


def ttlocate(seqx):
    aa = ['R', 'K', 'Q', 'E', 'D', 'N', 'Y', 'P', 'T', 'S', 'H', 'A', 'G', 'W', 'M', 'F', 'L', 'V', 'I', 'C']
    row_i1 = aa.index(seqx[0])
    row_i2 = aa.index(seqx[2])
    col_i1 = aa.index(seqx[1])
    col_i2 = aa.index(seqx[3])
    row_i = (row_i1 * 20) + row_i2
    col_i = (col_i1 * 20) + col_i2
    return (row_i, col_i)


# function to get 20x20 matrix given array of aa sequences and DC label
def ttmatrix(seqs, dc):
    # define empty matrix to fill in
    ttmat = np.full((400, 400), np.nan)
    for i, s in enumerate(seqs):
        # ttmat[ttlocate(s)[0],ttlocate(s)[1]] = dc[i]
        ttmat[ttlocate(s)] = dc[i]
    mask = np.isnan(ttmat)
    return ttmat, mask
import xgboost as xgb

# plot the ttmatrix, and highlight the query if present
def ttplot(seqs, dc, query=[]):
    """Plots a 20x20 Matrix based on provided Sequences and LogDC values. The query is optional and draws a yellow square around the queried sequence

    Args:
        seqs ([String]): array of sequences to build the ttplot with
        dc ([Float]): array of log DC values correspondingdescription to the sequences.
        query (list, optional): List of query sequences to see where they might show up on the 20x20 Plot. Defaults to [].

    Returns:
        [type]: [description]
    """
    ttmat, mask = ttmatrix(seqs, dc)

    aa = ['R', 'K', 'Q', 'E', 'D', 'N', 'Y', 'P', 'T', 'S', 'H', 'A', 'G', 'W', 'M', 'F', 'L', 'V', 'I', 'C']
    ticks = [[a[0]] + [''] * 19 for a in aa]
    ticks = [j for i in ticks for j in i]
    sns.set(rc={'figure.figsize': (9, 6)})

    # if dc values are binary then use simple color map
    cm = 'Blues' if len(np.unique(dc)) == 2 else 'coolwarm'
    ax = sns.heatmap(ttmat, cmap=cm, xticklabels=ticks, yticklabels=ticks, mask=mask)
    ax.set_title('Position of queried sequence/s in 20x20 plot (log (deep conversion))')

    if len(query) > 0:
        for q in query:
            query_pos = ttlocate(q)[1], ttlocate(q)[0]
            ax.add_patch(Rectangle(query_pos, 1, 1, fill=True, edgecolor='yellow', lw=8))

    return ax.figure


def evalplots(y_test, y_score, y_pred, labels, name_modifier):
    precision, recall, thr = precision_recall_curve(y_test, y_score)
    average_precision = average_precision_score(y_test, y_score)
    f1score = f1_score(y_test, y_pred)
    f1vec = [hmean([precision[i], recall[i]]) for i in range(sum(recall != 0))]

    # plt.plot([i/len(f1vec) for i in range(len(f1vec))],f1vec,color='r',alpha=0.2)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}, F1={1:0.2f}'.format(average_precision, f1score))
    plt.tight_layout()
    plt.savefig(f'figures/precision_recall_{name_modifier}.svg')
    plt.savefig(f'figures/precision_recall_{name_modifier}.png', dpi=300)
    plt.show()


    plt.step(thr[recall[:-1] != 0], f1vec, color='r', alpha=0.2, where='post')
    plt.fill_between(thr[recall[:-1] != 0], f1vec, step='post', alpha=0.2, color='r')
    plt.xlabel('Threshold')
    plt.ylabel('Estimated F1-Scores')
    plt.ylim([0.0, 1.0])
    plt.axvline(x=0.5, color='r')
    plt.title('Threshold Vs F1-Score: Max F1 ={0:0.2f}, Reported F1={1:0.2f}'.format(np.max(f1vec), f1score))
    plt.tight_layout()
    plt.savefig(f'figures/threshold_f1_{name_modifier}.svg')
    plt.savefig(f'figures/threshold_f1_{name_modifier}.png', dpi=300)
    plt.show()



    cm = confusion_matrix(y_test, y_pred, labels)
    print('Recall: {0:0.2f}'.format(recall_score(y_test, y_pred)))
    print('Precision: {0:0.2f}'.format(precision_score(y_test, y_pred)))
    display(pd.DataFrame(cm, columns=['Negative', 'Positive'], index=['Negative', 'Positive']))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap='hot')
    print('\n')
    plt.title('Confusion matrix : Acc={0:0.2f}'.format(accuracy_score(y_test, y_pred)))
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    print('--------------------------------------------------------')
