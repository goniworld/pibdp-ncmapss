"""Fig. 11 (WS1 learning curves) and Fig. 12 (WS2 semi-synthetic study) from committed results.
Usage: python make_fig11_fig12.py <addendum_v1.2 dir> <output dir>"""
import sys, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
R=sys.argv[1]; OUT=sys.argv[2]; import os; os.makedirs(OUT, exist_ok=True)
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'legend.frameon': False,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#8a8984', 'axes.linewidth': 0.6,
    'axes.grid': True, 'grid.color': '#e6e5e0', 'grid.linewidth': 0.5, 'axes.axisbelow': True,
    'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.labelcolor': '#0b0b0b',
    'lines.linewidth': 1.5, 'savefig.dpi': 300, 'figure.dpi': 110})
COL = dict(v0='#2a78d6', gb='#eb6834', lstm='#1baf7a', rf='#eda100', gray='#b4b3ad', ink='#0b0b0b', ink2='#52514e', purple='#7d5bc4')
def save(fig,name): fig.savefig(f'{OUT}/{name}.png',bbox_inches='tight'); fig.savefig(f'{OUT}/{name}.pdf',bbox_inches='tight')
# ---- Fig 11
C=pd.read_csv(f'{R}/ws1/ws1_curve.csv',dtype={'m':str}).set_index('m'); P=pd.read_csv(f'{R}/ws1/ws1_pool.csv')
x=np.arange(4); lab=['2\n(55 engines)','3\n(47)','5\n(18, 2 subsets)','all\n(55)']
fig,axs=plt.subplots(1,2,figsize=(7.0,2.7),sharey=True)
ax=axs[0]
for col,name,c,mk in [('V0','PI-BDP V0',COL['v0'],'o'),('TSMixer','TSMixer',COL['lstm'],'s'),('TCN','TCN',COL['rf'],'^'),('GB','GB',COL['gb'],'D')]:
    ax.plot(x,C[col].values,color=c,marker=mk,ms=4,label=name,lw=1.4 if col=='V0' else 1.1)
pool=float(P.loc[P.model=='TSMixer S-pool','rmse'].iloc[0])
ax.plot([0],[pool],marker='*',ms=9,color=COL['lstm'],mec='white',ls='none',label='TSMixer, S-pool (m = 2)')
ax.set_title('(a) PI-BDP vs data-driven models',loc='left'); ax.set_ylabel('Window RMSE (cycles)')
ax.set_xticks(x,lab); ax.set_xlabel('Failed training engines m (held-out engines)'); ax.legend(loc='upper right')
ax=axs[1]
for col,name,c,ls in [('V0','V0 (product weights)',COL['v0'],'-'),('noWeibull','noWeibull (β = 0)',COL['ink2'],'--'),('meld0.5','Melding β = 0.5',COL['purple'],'-'),('meld1','Melding β = 1',COL['purple'],':')]:
    ax.plot(x,C[col].values,color=c,ls=ls,marker='o',ms=3.5,label=name,lw=1.3)
ax.set_title('(b) PI-BDP re-weighting variants',loc='left'); ax.set_xticks(x,lab); ax.set_xlabel('Failed training engines m (held-out engines)'); ax.legend(loc='upper right')
axs[0].set_ylim(3,7.2)
save(fig,'Fig11_learning_curves')
# ---- Fig 12
T=pd.read_csv(f'{R}/ws2/ws2_table.csv')
ms=[2,3,5,10,20]
meth=[('V0','V0 (product)',COL['v0'],'-','o'),('meld0.5','Melding β = 0.5',COL['purple'],'-','s'),('meld1','Melding β = 1',COL['purple'],':','s'),('noWeibull','noWeibull (β = 0)',COL['ink2'],'--','^')]
scen=[('G0','G0 correct model'),('G1','G1 wrong path shape'),('G2','G2 doubled noise'),('G3','G3 censored training')]
fig,axs=plt.subplots(2,4,figsize=(7.2,3.8),sharex=True)
for j,(g,title) in enumerate(scen):
    for i,(met,ylab) in enumerate([('cov90','Raw 90% coverage'),('w1','Wasserstein distance\nto oracle (cycles)')]):
        ax=axs[i,j]
        for key,name,c,ls,mk in meth:
            d=T[(T.scen==g)&(T.method==key)].set_index('m').reindex(ms)
            ax.plot(ms,d[met].values,color=c,ls=ls,marker=mk,ms=3,lw=1.2,label=name)
        if met=='cov90':
            o=T[(T.scen==g)&(T.method=='oracle')].set_index('m').reindex(ms)[met]
            ax.plot(ms,o.values,color=COL['gray'],lw=1.0,ls='-',label='Oracle'); ax.set_ylim(0.25,0.95)
            ax.set_title(title,loc='left')
        ax.set_xscale('log'); ax.set_xticks(ms,[str(v) for v in ms]); ax.minorticks_off()
        if j==0: ax.set_ylabel(ylab)
        if i==1: ax.set_xlabel('Training engines m')
h,l=axs[0,0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=5,bbox_to_anchor=(0.5,-0.04))
fig.tight_layout(rect=(0,0.05,1,1))
save(fig,'Fig12_semisynthetic')
print('ok')
