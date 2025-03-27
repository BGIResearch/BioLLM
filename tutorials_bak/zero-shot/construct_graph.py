#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :construct_graph
# @Time      :2024/6/18 17:20
# @Author    :Luni Hu

import json
import os.path

# Load the JSON file
with open('/home/share/huadjyin/home/s_huluni/project/bio_model/biollm/case/models/scmamba/vocab.json', 'r') as f:
    vocab = json.load(f)

import pyarrow.feather as feather
import pandas as pd

df = feather.read_feather('/home/share/huadjyin/home/s_huluni/project/bio_model/biollm/case/ref/hg19-tss-centered-10kb-10species.mc9nr.genes_vs_motifs.rankings.feather')


motif_df = pd.read_csv("/home/share/huadjyin/home/s_huluni/project/bio_model/biollm/case/ref/motifs-v9-nr.hgnc-m0.001-o0.0.tbl", sep='\t', index_col=0, low_memory=False)

df.index = df["motifs"].tolist()
df = df.drop("motifs", axis=1)

motif_dict = dict(zip(motif_df.index, motif_df["gene_name"]))
df = df.loc[df.index.isin(motif_dict.keys()), :]
df.index = [motif_dict[key] for key in df.index]

df = df.loc[df.index.isin(list(vocab.keys())), df.columns.isin(list(vocab.keys()))]

from tqdm import tqdm


def process_edges(df):
    pos_edges = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing TFs"):
        pos_edges.append(row.sort_values(ascending=True)[:50].index.tolist())

    pos_edges = pd.DataFrame(pos_edges, index=df.index)
    pos_edges = pos_edges.melt(ignore_index=False)
    pos_edges["Gene1"] = pos_edges.index.tolist()
    pos_edges["Gene2"] = pos_edges["value"].tolist()
    pos_edges = pos_edges.loc[:, ["Gene1", "Gene2"]]

    return pos_edges

edgeDF = process_edges(df)

import networkx as nx
g = nx.from_pandas_edgelist(edgeDF, source="Gene1", target="Gene2", create_using=nx.DiGraph)
print(len(g.edges))
g.remove_edges_from(nx.selfloop_edges(g))
print(len(g.edges))

def remove_cycle(graph):

    while True:

        try:
            cycle = nx.find_cycle(graph, orientation='original')
            print("Found cycle:", cycle)
            for u, v, _ in cycle:
                graph.remove_edge(u, v)

        except nx.exception.NetworkXNoCycle:
            print("No more cycles detected.")
            break

    return graph

g = remove_cycle(g)
print(len(g.edges))

edge_path = os.path.join("/home/share/huadjyin/home/s_huluni/project/bio_model/biollm/case/graph",
                          "tf_target_edgelist.csv")
edge_df = nx.to_pandas_edgelist(g)
edge_df.to_csv(edge_path, index=False)