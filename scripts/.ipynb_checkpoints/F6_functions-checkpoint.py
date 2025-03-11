import os
from os.path import join
import numpy as np
import pickle
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
#import matplotlib
from scipy import stats
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')
sns.set_context("paper")

CL={'Th':'#F45A25','ACA':'#008080','RSC':'#9F00E7','CTRL':'#1A202E'} #Th, ACA, RSC, CTRL

colors=np.array(['#F45A25','#008080','#9F00E7','#1A202E'])

ordered_list=['Th','ACA','RSC','CTRL']


def bars(p, bottom, top, top_adjust):
    # Get info about y-axis
    yrange = top - bottom

    # Columns corresponding to the datasets of interest
    x1 = 1
    x2 = 2
    # What level is this bar among the bars above the plot?
    level = 1
    # Plot the bar
    bar_height = (yrange * 0.08 * level) + top+ top_adjust
    bar_tips = bar_height - (yrange * 0.05)
    plt.plot(
        [x1, x1, x2, x2],
        [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
    # Significance level

    if p < 0.001:
        sig_symbol = '***'
    elif p < 0.01:
        sig_symbol = '**'
    elif p < 0.05:
        sig_symbol = '*'
    else:
        sig_symbol = '$^{n.s}$'
    text_height = bar_height + (yrange * 0.025)
    plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k',fontsize=14)


et=['RSC','RSC','ACAv','Th']


def make_df(kind="1"):
    """
    make dataframe with 
    kind = 1: (pre-post) of th values
    kind = 2: (pre-post)/pre of eta values
    kind = 3: (pre-post)/(pre+post) of th values
    """

    df = pd.read_csv(join('../data/SBI/6eta_1G_Inference/df235_prepost.csv'), sep='\t', encoding='utf-8')
    etas = ['eta_DMN', 'eta_Vis','eta_LCN', 'eta_BF', 'eta_HPF', 'eta_Th']
    #  filter df with with prepost=pre and sort by tag
    df_pre = df[df['prepost'] == 'pre'].sort_values(by=['tag']).reset_index(drop=True)
    df_post = df[df['prepost'] == 'post'].sort_values(by=['tag']).reset_index(drop=True)
    
    df_pre['group'] = pd.Categorical(df_pre['group'], ordered_list, ordered=True)
    df_pre=df_pre.sort_values('group').reset_index(drop=True)
    df_post['group'] = pd.Categorical(df_post['group'], ordered_list, ordered=True)
    df_post=df_post.sort_values('group').reset_index(drop=True)
    
    # make a df and difference th values in columns start with eta
    df1 = pd.DataFrame()
    df1['tag'] = df_pre['tag']
    df1['group'] = df_pre['group']
    df1['n_subj'] = df_pre['n_subj']
    if kind == "1":
        for e in etas:
            df1[e] = (df_pre[e] - df_post[e])
    elif kind == "2":
        for e in etas:
            df1[e] = (df_pre[e] - df_post[e]) / df_pre[e]
    elif kind == "3":
        for e in etas:
            df1[e] = (df_pre[e] - df_post[e]) / (df_pre[e] + df_post[e])
    
    assert (df_pre['tag'].values == df_post['tag'].values).all()
    assert (df_pre['group'].values == df_post['group'].values).all()
    assert (df_pre['n_subj'].values == df_post['n_subj'].values).all()

    return df1

def wrapper_vis(df, fig, ax, title=""):
    labels = ["Th", "ACA", "RSC", "CTRL"]
    X = df.drop(columns=["group", 'tag', 'n_subj'])
    y = df['group']

    
    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(X)
    yi = np.where(y == "Th", 0, np.where(y == "ACA", 1, np.where(y == "RSC", 2, 3)))
    print(X_2d[:, 0].shape)
    #ax[0].grid(True)
    cax = ax[0].scatter(X_2d[:, 0], X_2d[:, 1], s=50, c=np.repeat(colors,30), alpha=0.6)
    ax[0].set_xlabel('Dim 1', fontsize=12)
    ax[0].set_ylabel('Dim 2', fontsize=12)
    #cbar = fig.colorbar(cax, ticks=range(4), ax=ax[0])
    #cbar.set_ticklabels(labels)
    #ax[0].set_title(title)
    classif = RandomForestClassifier
    scores = cv_classif_wrapper(classif(), X, yi, n_splits=5, verbose=1, weighted=1, random_state=42)

    model = classif(random_state=42)
    y_pred = cross_val_predict(model, X, yi, cv=5)
    cm = confusion_matrix(yi, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.from_predictions(yi, y_pred, ax=ax[1], colorbar=False, normalize='true',
                        cmap=plt.cm.Blues, display_labels=labels);

    plt.tight_layout()
    return fig, ax

def cv_classif_wrapper(classifier, X, y, n_splits=5, random_state=42, verbose=0, weighted=False):
    '''
    cross validation wrapper
    '''
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=random_state)

    if not weighted:
        scores = cross_validate(classifier, X, y, cv=cv, scoring=[
                                'f1', 'accuracy', 'recall', 'precision'])
    else:
        scores = cross_validate(classifier, X, y, cv=cv, scoring=[
                                'f1_weighted', 'accuracy', 'recall_weighted', 'precision_weighted'])

    if verbose:
        print(f"=====================")
        print(f"Classifier: {classifier.__class__.__name__}")
        if not weighted:

            print(f"Accuracy:    {scores['test_accuracy'].mean():.3f} (+/- {scores['test_accuracy'].std()*2:.3f})")
            print(f"Recall:      {scores['test_recall'].mean():.3f} (+/- {scores['test_recall'].std()*2:.3f})")
            print(f"Precision:   {scores['test_precision'].mean():.3f} (+/- {scores['test_precision'].std()*2:.3f})")
            print(f"F1:          {scores['test_f1'].mean():.3f} (+/- {scores['test_f1'].std()*2:.3f})")
        else:
            print(f"Accuracy:    {scores['test_accuracy'].mean():.3f} (+/- {scores['test_accuracy'].std()*2:.3f})")
            print(f"Recall:      {scores['test_recall_weighted'].mean():.3f} (+/- {scores['test_recall_weighted'].std()*2:.3f})")
            print(f"Precision:   {scores['test_precision_weighted'].mean():.3f} (+/- {scores['test_precision_weighted'].std()*2:.3f})")
            print(f"F1:          {scores['test_f1_weighted'].mean():.3f} (+/- {scores['test_f1_weighted'].std()*2:.3f})")
        print(f"=====================")

    return scores

