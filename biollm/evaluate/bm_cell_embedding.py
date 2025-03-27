#!/usr/bin/env python3
# coding: utf-8
"""
@author: Ping Qiu  qiuping1@genomics.cn
@last modified by: Ping Qiu
@file: bm_cell_embedding.py
@time: 2025/3/12 10:11
"""

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score


def cluster_metrics(X, y_pred, y_true):
    """
    embedding 是 X，聚类标签是 y_pred，真实细胞类型是 y_true
    """
    silhouette = silhouette_score(X, y_pred)
    ch_index = calinski_harabasz_score(X, y_pred)
    db_index = davies_bouldin_score(X, y_pred)

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)

    print(f"Silhouette Score: {silhouette} |衡量不同聚类的分布情况，越高越好|")
    print(f"CH Index: {ch_index}  |计算簇间方差与簇内方差的比值，值越大表示聚类效果越好|")
    print(f"DB Index: {db_index}  |衡量每个簇的紧密度和簇间的可分性，值越小越好|")
    print(f"ARI: {ari} |衡量两个分区的一致性，去除随机性影响，范围 [−1,1]，越接近 1 越好|")
    print(f"NMI: {nmi}  |计算聚类与真实标签的信息增益，范围 [0,1]，越高越好|")
    print(f"AMI: {ami}  |类似于 NMI，但去除了随机因素影响，范围 [0,1]，越高越好|")
    return {'silhouette': silhouette, 'ch_index': ch_index, 'db_index': db_index, 'ari': ari, 'nmi': nmi, 'ami': ami}


def batch_effect_metrics():
    pass
