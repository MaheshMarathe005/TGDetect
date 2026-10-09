"""Feature-only weak-label controls, no model/feature selection based on test."""
from pathlib import Path
import time,json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from run_causal_pilot import metrics,select_threshold
from windows_apt_data import write_json,digest
root=Path('artifacts/windows_apt_v1');path=root/'prepared/data.npz';a=np.load(path,allow_pickle=False)
x,y=a['x'],a['y'];tr=int(a['train_end']);ve=int(a['val_end']);results={}
for name,model in [('logistic',LogisticRegression(max_iter=1000,random_state=0)),('hist_gradient_boosting',HistGradientBoostingClassifier(max_iter=100,early_stopping=False,random_state=0))]:
 start=time.perf_counter();model.fit(x[:tr],y[:tr]);p=model.predict_proba(x)[:,1];threshold=select_threshold(y[tr:ve],p[tr:ve])
 results[name]={'seconds':time.perf_counter()-start,'threshold':threshold,'validation':metrics(y[tr:ve],p[tr:ve],threshold),'test':metrics(y[ve:],p[ve:],threshold)}
 np.savez_compressed(root/f'control_{name}_predictions.npz',probability=p,y=y,row_ids=a['row_ids'])
write_json(root/'feature_controls.json',{'label_scope':'WEAK_LABEL_WAZUH_MITRE_PRESENCE','data_sha256':digest(path),'results':results,'warning':'Only context for annotation prediction; no attack-ground-truth claim.'})
print('Feature-only controls completed; scores saved separately from neural training.')
