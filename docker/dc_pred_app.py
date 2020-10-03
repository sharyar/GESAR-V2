#!/usr/bin/env python
import dash
import dash_table as dt
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from plotly.tools import mpl_to_plotly

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from matplotlib.patches import Rectangle
import pickle
import re
from io import BytesIO
import base64


#feature extractor contains information to create features for a given query sequence
feat_extdict  = pickle.load(open("feature_extractor_dict_cor.pickle", "rb"))
#load experimental DC values for 20x20 plot
logDC=pickle.load(open('logDC.pickle','rb'))
#load trained xgb models
mod_low  = pickle.load(open("mod_low_xgb_cor.pickle", "rb"))
mod_high  = pickle.load(open("mod_high_xgb_cor.pickle", "rb"))


#this function creates featuers for a given query sequence
def feat_extractor(seqs, ft_dict):
    #seqs: array of sequences to extract features from
    #ft_dict: feature extraction dict
    
    ext_feat = dict()
    for seq_ex in seqs:
        ext_feat[seq_ex] = dict()
        for pos,b in enumerate(seq_ex):
            for featn in range(8):
                ext_feat[seq_ex]['vhse'+str(pos+1)+'.'+str(featn+1)] = ft_dict['vhse'][featn+1][b]
            for featn in range(3):
                ext_feat[seq_ex]['z'+str(pos+1)+'.'+str(featn+1)] = ft_dict['z'][featn+1][b]

        for pat in feat_extdict['pat_feat20']:
            ext_feat[seq_ex][pat] = int(pd.notnull(re.search(pat,seq_ex)))
    
    featext_df = pd.DataFrame(ext_feat).T
    return(featext_df.loc[:,ft_dict['Xpvz_names']])

###code for creating 20x20 plot
#function to locate index position of seq in 20x20 plot
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
    ttmat = np.zeros((400,400))
    for i,s in enumerate(seqs):
        #ttmat[ttlocate(s)[0],ttlocate(s)[1]] = dc[i]
        ttmat[ttlocate(s)] = dc[i]
    return ttmat   


#plot the ttmatrix, and highlight the query if present
def ttplot(seqs,dc,query = []):
    ttmat = ttmatrix(seqs,dc)

    aa = ['R','K','Q','E','D','N','Y','P','T','S','H','A','G','W','M','F','L','V','I','C']
    ticks = [[a[0]] + ['']*19 for a in aa]
    ticks = [j for i in ticks for j in i]
    sns.set(rc={'figure.figsize':(9,6)})
    
    #if dc values are binary then use simple color map
    cm = 'Blues' if len(np.unique(dc)) ==2 else 'RdBu_r'        
    ax = sns.heatmap(ttmat, cmap=cm,xticklabels=ticks, yticklabels=ticks)
    ax.set_title('Position of queried sequence/s in 20x20 plot (log (deep conversion))')

    if len(query)>0:
        for q in query:
            query_pos = ttlocate(q)[1], ttlocate(q)[0]
            ax.add_patch(Rectangle(query_pos, 1, 1, fill=True,edgecolor='yellow', lw=8))
    
    return ax.figure

def fig_to_uri(in_fig, close_all=True, **save_args):
    # type: (plt.Figure) -> str
    """
    Save a figure as a URI
    :param in_fig:
    :return:
    """
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)

link_text = '''[FAQ](https://docs.google.com/document/d/1UhMGYL-1bqw6nDpTzhFDizWx9GqKyaXXijQwNMwgq3Q/edit?usp=sharing)'''

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H3(children='DC Prediction App'),

    html.H5(children='''
        Machine learned model to predict probability of low or high deep conversion values.
    '''),
    
    html.Div(children='''
        Enter amino acid tetramer sequences below. eg: TWWY
    '''),
    
    html.Div(children='''
        Mutliple sequences can be entered seperated by comma. eg: PWCC,PRGS,WLQI,NPYK
    '''),
    
    html.Div([dcc.Markdown(children=link_text)]),

    dcc.Input(id='input-state', type='text', value=''),
    html.Button(id='submit-button', n_clicks=0, children='Submit'),
    html.Div(id='plot_div'),
    html.Div(id='table-desc'),
    html.Div(id='table-container'),
    
])


#@app.callback([Output('output-state', 'children'),
#              Output('table-container', 'children')],
#              [Input('submit-button', 'n_clicks')],
#              [State('input-state', 'value')])
@app.callback([Output('plot_div', 'children'), 
               Output('table-container', 'children'),
              Output('table-desc', 'children')],
              [Input('submit-button', 'n_clicks')],
              [State('input-state', 'value')])
def update_output(n_clicks,input1):
    if n_clicks:
        input1 = input1.replace(" ", "")
        query = input1.split(',')
        feats =  feat_extractor(query,feat_extdict)
        low_probs = mod_low.predict_proba(feats.values)[:,1] *100
        high_probs = mod_high.predict_proba(feats.values)[:,1]*100
        query = list(feats.index) #removes duplicates if any and maintains order
        fig = ttplot(logDC.index,logDC['log.label'],query)
        out_url = fig_to_uri(fig)

        #print("XGBoost probability for highest and lowest 5 percentile of deep conversion values: " )
        probs_df = pd.DataFrame([query,list(high_probs),list(low_probs)]).T
        probs_df.columns = ['Query','HIGH','LOW']
        probs_df['HIGH']=probs_df['HIGH'].map('{:,.2f}%'.format)
        probs_df['LOW']=probs_df['LOW'].map('{:,.2f}%'.format)
        probs_dt = dt.DataTable(
            data=probs_df.to_dict('records'),
            columns=[{"name": i, "id": i,} for i in (probs_df.columns)],
            style_header={'backgroundColor': 'blue','fontSize':18, 'font-family':'sans-serif'},
            style_cell={'backgroundColor': 'rgb(50, 50, 50)','color': 'white', 'fontSize':18, 'font-family':'sans-serif',
                       'minWidth': '180px', 'width': '180px', 'maxWidth': '180px'},
        ) 
        tab_desc = '''XGBoost probability for highest and lowest 5 percentile of deep conversion values: '''
        return (html.Img(id = 'cur_plot', src = out_url), probs_dt,tab_desc)

if __name__ == '__main__':
    app.run_server(host='0.0.0.0',debug=False, port=8050)