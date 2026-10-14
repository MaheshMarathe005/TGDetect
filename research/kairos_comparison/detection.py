"""KAIROS paper §4.3 queues and §4.4 weighted Louvain investigation.

Queue products are evaluated in log space. Online and retrospective scores are
separate: a later alert can flag an earlier window only in retrospective output.
"""
from collections import Counter
import math
import numpy as np


def fit_rareness(src, dst, window):
    counts = Counter()
    windows = np.unique(window)
    if len(windows) < 2:
        raise ValueError('At least two benign training windows needed for IDF')
    _, starts, sizes = np.unique(window, return_index=True, return_counts=True)
    for lo, size in zip(starts, sizes):
        counts.update(set(map(int, np.r_[src[lo:lo+size], dst[lo:lo+size]])))
    n = len(windows)
    return dict(counts=counts, n=n, alpha=math.log(.9 * n))


def detect(src, dst, windows, losses, rareness):
    queues, rows = [], []
    unique, starts, sizes = np.unique(windows, return_index=True, return_counts=True)
    for wid, lo, size in zip(unique, starts, sizes):
        ix = np.arange(lo, lo + size)
        re = losses[ix]
        sigma = float(re.mean() + 1.5 * re.std())
        high = ix[re > sigma]
        anomalous_nodes = set(map(int, np.r_[src[high], dst[high]]))
        rare = {v for v in anomalous_nodes
                if math.log(rareness['n'] / (rareness['counts'].get(v, 0) + 1)) > rareness['alpha']}
        log_score = float(np.log(losses[high].mean())) if len(high) else -float('inf')
        row = dict(window=int(wid), sigma=sigma, anomalous_edges=high.tolist(),
                   suspicious_nodes=sorted(rare), log_window_score=log_score,
                   online_score=log_score, retrospective_score=log_score)
        index = len(rows)
        rows.append(row)
        matches = [q for q in queues if q['nodes'] & rare]
        if not matches:
            queues.append(dict(nodes=set(rare), members=[index], score=log_score))
        else:
            for q in matches:
                q['nodes'].update(rare)
                q['members'].append(index)
                q['score'] += log_score
                row['online_score'] = max(row['online_score'], q['score'])
                for previous in q['members']:
                    rows[previous]['retrospective_score'] = max(rows[previous]['retrospective_score'], q['score'])
    return rows


def summaries(src, dst, relation, losses, rows, log_threshold, seed=0):
    import networkx as nx
    graph = nx.Graph()
    # Keep directed typed records for export; only community optimization is undirected.
    reduced = {}
    for row in rows:
        if row['retrospective_score'] <= log_threshold:
            continue
        for i in row['anomalous_edges']:
            key = (int(src[i]), int(dst[i]), int(relation[i]))
            old = reduced.setdefault(key, dict(count=0, weight=0.))
            old['count'] += 1
            old['weight'] = max(old['weight'], float(losses[i]))
    for (s, d, rel), data in reduced.items():
        weight = graph.get_edge_data(s, d, {}).get('weight', 0.) + data['weight']
        graph.add_edge(s, d, weight=weight)
    if not graph.number_of_edges():
        return []
    communities = nx.community.louvain_communities(graph, weight='weight', seed=seed)
    return [dict(nodes=sorted(c), edges=[dict(src=s, dst=d, relation=r, **v)
        for (s, d, r), v in reduced.items() if s in c and d in c]) for c in communities]
