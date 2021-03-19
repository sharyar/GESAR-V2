from collections import defaultdict
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

def ttlocate(seqx):
    aa = ['R','K','Q','E','D','N','Y','P','T','S','H','A','G','W','M','F','L','V','I','C']
    row_i1 = aa.index(seqx[0])
    row_i2 = aa.index(seqx[2])
    col_i1 = aa.index(seqx[1])
    col_i2 = aa.index(seqx[3])
    row_i = (row_i1*20) + row_i2
    col_i = (col_i1*20) + col_i2
    return (row_i,col_i)

#function to get 20x20 matrix given array of aa sequences and DC label
def ttmatrix(seqs,dc):
    #define empty matrix to fill in
    ttmat = np.full((400,400), np.nan)
    for i,s in enumerate(seqs):
        #ttmat[ttlocate(s)[0],ttlocate(s)[1]] = dc[i]
        ttmat[ttlocate(s)] = dc[i]
    mask = np.isnan(ttmat)
    return ttmat, mask   


#plot the ttmatrix, and highlight the query if present
def ttplot(seqs,dc,query = []):
    """Plots a 20x20 Matrix based on provided Sequences and LogDC values. The query is optional and draws a yellow square around the queried sequence

    Args:
        seqs ([String]): array of sequences to build the ttplot with
        dc ([Float]): array of log DC values correspondingdescription to the sequences. 
        query (list, optional): List of query sequences to see where they might show up on the 20x20 Plot. Defaults to [].

    Returns:
        [type]: [description]
    """
    ttmat, mask = ttmatrix(seqs,dc)

    aa = ['R','K','Q','E','D','N','Y','P','T','S','H','A','G','W','M','F','L','V','I','C']
    ticks = [[a[0]] + ['']*19 for a in aa]
    ticks = [j for i in ticks for j in i]
    sns.set(rc={'figure.figsize':(9,6)})
    
    #if dc values are binary then use simple color map
    cm = 'Blues' if len(np.unique(dc)) ==2 else 'coolwarm'        
    ax = sns.heatmap(ttmat, cmap=cm,xticklabels=ticks, yticklabels=ticks, mask=mask)
    ax.set_title('Position of queried sequence/s in 20x20 plot (log (deep conversion))')

    if len(query)>0:
        for q in query:
            query_pos = ttlocate(q)[1], ttlocate(q)[0]
            ax.add_patch(Rectangle(query_pos, 1, 1, fill=True,edgecolor='yellow', lw=8))
    
    return ax.figure


def hist_logdc_plot(dfs, labels, colors=['blue', 'red', 'gray'], title='Histogram of LogDC values'):
    """Plots histograms of provided dataframes with corresponding means. 

    Args:
        dfs ([pd.Dataframe]): List/array of dataframes
        labels ([String]): List/array of strings describing the dataframes
        colors (list, optional): Colors to use for the histograms. Defaults to ['blue', 'red', 'gray'].
    """
    fig, axes = plt.subplots(1,1, figsize=(14,6))
    labels_with_mean = []
    
    plots = []
    
    for n, df in enumerate(dfs):
        plots.append(sns.histplot(df['log.label'], color=colors[n], axes=axes, alpha=0.7, kde=True, legend=True))
        plt.axvline(df['log.label'].mean(), axes=axes, color=colors[n], linestyle='dashed')
        labels_with_mean.append(str(f"{labels[n]} mean: {df['log.label'].mean():0.4f} +- {df['log.label'].std():0.4f}"))
        labels_with_mean.append(None)
    plt.legend([x for x in labels_with_mean])
    plt.title(title)
    plt.tight_layout()
    
    return fig

def aa_count(df, normalize):
    """
    Takes a given dataframe and counts the number of occurences of aa within the dataframe. Creates a dictionary with the structure AA:Count
    """
    aa_list = ['R','K','Q','E','D','N','Y','P','T','S','H','A','G','M','F','L','V','I','C','W']
    aa_counter_dict = defaultdict(int)
    for seq in df.seq.values:
        for a in aa_list:
            aa_counter_dict[a] += seq.count(a)

    if normalize:
        return normalize_counter(aa_counter_dict)
    else:
        return aa_counter_dict

def base_normalize_aa_count(df_normalized_count, base_df_normalized_count):
    base_normalized_counter_dict = defaultdict(int)

    for k,v in df_normalized_count.items():
        base_normalized_counter_dict[k] = df_normalized_count[k] - base_df_normalized_count[k]

    return base_normalized_counter_dict

def aa_distrib_plotter(dfs, captions, colors=['orange', 'olive', 'gray'], 
                       normalize=False, normalize_base=False, base_df=None):
    """
    Takes a group of DFs and captions to create plots of their aa distributions. Can specify colors optionally via the colors argument. Can also specify if the plot should be normalized. Max 3 plots at a time. To normalize to a base, set normalize_base to True and give a base df to normalize against. 
    """
    num_df = len(dfs)
    fig, axes = plt.subplots(1, num_df, figsize=(8*num_df, 6), sharey = True)
    
    for n, df in enumerate(dfs):

        if normalize_base or base_df != None:
            base_aa_count = aa_count(base_df, normalize=True)
            target_aa_count = aa_count(df, normalize=True)
            aa_counter_dict = experimental_base_normalize_aa_count(target_aa_count, base_aa_count)

        else:
            aa_counter_dict = aa_count(df, normalize=normalize)

        if len(dfs) == 1:
            axes.bar(*zip(*aa_counter_dict.items()), color=colors[n])
            axes.set_ylabel('Number of Occurences of AA')
            axes.set_title(captions[n])

        else:
            axes[n].bar(*zip(*aa_counter_dict.items()), color=colors[n])
            axes[n].set_ylabel('Number of Occurences of AA')
            axes[n].set_title(captions[n])

    return fig

def experimental_base_normalize_aa_count(df_normalized_count, base_df_normalized_count):
    base_normalized_counter_dict = defaultdict(int)

    for k,v in df_normalized_count.items():
        base_normalized_counter_dict[k] = (df_normalized_count[k] - base_df_normalized_count[k])/base_df_normalized_count[k]

    return base_normalized_counter_dict

def normalize_counter(counter_dict):
    """
    Takes a counter dictionary with a structure of AA:Count and normalizes it so count/total_counts. 
    Returns the normalized dict
    """
    total_aa = np.sum(list(counter_dict.values()))
    normalized_dict = {k:v/total_aa for (k, v) in counter_dict.items()}
    
    return normalized_dict

def descript_stats(dataframes):
    """Generates descriptive stats including mean, std, min and max for all given dataframes. 

    Args:
        dataframes ([pandas.Dataframe]): Provide a list or array of Dataframes for which you want descript stats. 
    """
    for df in dataframes:
        print('*'*50)
        mean = df['log.label'].mean()
        std = df['log.label'].std()
        minimum = df['log.label'].min()
        maximum = df['log.label'].max()
        print(f'{mean:.4f} +- {std:.4f}\nMin: {minimum:.4f}\nMaximum: {maximum:.4f}')

def filter_data(dfvz):
    """Returns an index of filtered rows based on preset criteria. In this case the critera are: 
    -Remove sequences containing C
    -Remove sequences that follow this pattern: PXXX but not PPXXX. 

    Args:
        dfvz ([pandas.Dataframe]): Dataframe to filter values from

    Returns:
        [ints]: indices of rows that are filtered. Should be used with df.loc to actually get the correct rows. 
    """
    # Filter Includes All Samples Containing Cysteine
    filter_cys = dfvz['seq'].str.contains('C')

    # Filter Includes All Samples Containing Proline in First Position but not in Second Position
    # I.e PCEQ is included but not PPEQ
    filter_proline = dfvz['seq'].str.contains('(^P[^P])')

    # Combined Filter to select any samples that satisfy either condition!
    filter_proline_cysteine = dfvz['seq'].str.contains('(^P[^P])|([C])')

    # Create index so it can be used for both DF20 and DFVZ (inverted to remove filtered rows)
    idx_filter_inverted = dfvz[~filter_proline_cysteine].index.to_list()

    print(f'Original Data Set Samples: {len(dfvz)}\
        \nSamples Containing Cystiene Residues: {len(dfvz[filter_cys])}\
        \nSamples Containing Proline (But Not Proline-Proline) {len(dfvz[filter_proline])}\
        \nTotal Samples that satisfy either condition (Some Overlap) {len(dfvz[filter_proline_cysteine])}\
        \nFinal Samples Used: {len(idx_filter_inverted)}\
        \nPercentage Removed: {len(idx_filter_inverted)/len(dfvz):.2%}')
    
    return idx_filter_inverted