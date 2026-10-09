"""Data-driven physical QA plots; no generated or reconstructed image data."""

import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
out=ROOT/'results/figures';out.mkdir(exist_ok=True)
profiles=json.loads((ROOT/'results/physical-profiles.json').read_text())
for region in P['regions']:
    meta=json.loads((ROOT/'cases'/region/'prepared.json').read_text())
    result=ROOT/'results'/region
    before=np.load(result/'coarse_CTX-0-k0-strict-100m-maps.npz')
    after=[np.load(result/f'original_frozen_v36-{seed}-k20-strict-100m-maps.npz') for seed in P['seeds']]
    valid=before['valid'];extent=[0,504*meta['dx_m']/1000,504*meta['dy_m']/1000,0]
    fig,axes=plt.subplots(2,3,figsize=(11,8),layout='constrained')
    for row,(key,refkey,label,cmap) in enumerate([('slope','reference_slope','Slope (degrees)','viridis'),('roughness','reference_roughness','Detrended roughness (m)','magma')]):
        fields=[before[refkey],before[key],np.mean([a[key] for a in after],axis=0)]
        vmax=max(float(np.max(z[valid])) for z in fields)
        for col,(z,title) in enumerate(zip(fields,['HiRISE reference','Before: coarse CTX','After: full K20, seed mean'])):
            ax=axes[row,col];im=ax.imshow(np.where(valid,z,np.nan),extent=extent,cmap=cmap,vmin=0,vmax=vmax)
            ax.set_title(title);ax.set_xlabel('Eastward distance (km)');ax.set_ylabel('Southward distance (km)')
        fig.colorbar(im,ax=list(axes[row]),label=label,shrink=.8)
    fig.suptitle(region.replace('_',' ').title()+' | fixed 100 m support, quality-screened')
    fig.savefig(out/f'{region}-physical-maps.png',dpi=180);plt.close(fig)
    feats=json.loads((ROOT/'cases'/region/'features.json').read_text())['features']
    ref=json.loads((result/'reference-profiles.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for idx,(ax,f) in enumerate(zip(axes.flat,feats)):
        mask=np.zeros(504,dtype=bool);mask[f['valid_indices']]=True
        distance=np.arange(504)*f['spacing_m']/1000
        ax.plot(distance,np.where(mask,ref[idx]['profile_values_m'],np.nan),color='#202124',lw=1.8,label='HiRISE')
        for method,k,color,label in [('coarse_CTX',0,'#8297A6','Coarse CTX'),('native_CTX',0,'#965B3F','Native CTX'),('original_frozen_v36',20,'#314858','Full K20')]:
            chosen=[r for r in profiles if r['region']==region and r['method']==method and r['k']==k and r['index']==f['index']]
            values=np.mean([r['profile_values_m'] for r in chosen],axis=0)
            ax.plot(distance,np.where(mask,values,np.nan),color=color,label=label,lw=1.3)
        if f['usable']:ax.axvline(f['reference_position_index']*f['spacing_m']/1000,color='#D94332',ls=':',lw=1)
        ax.set_title(f"{f['axis']} {f['index']} | "+('valid fixed profile' if f['usable'] else 'insufficient reference support'))
        ax.set_xlabel('Distance along profile (km)');ax.set_ylabel('Elevation (m)');ax.grid(alpha=.15)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle(region.replace('_',' ').title()+' | all prespecified profiles')
    fig.savefig(out/f'{region}-fixed-profiles.png',dpi=180);plt.close(fig)
print('PLOTS_DONE',flush=True)
