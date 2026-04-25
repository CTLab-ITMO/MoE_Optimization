#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import torch
from datetime import datetime
import os
import sys
import numpy as np
sys.path.append("..")
from common.mlp_utils import FUSION_METHODS, best_metrics

ACCURACY_DIR = "accuracy"

replace_types = ["seq", "single", "smart_seq"]
fusions = list(FUSION_METHODS.keys())
fusions += ["zero"]
fusions += [f"Best_{metric}" for metric in best_metrics.keys()]


colors = plt.cm.tab20(np.linspace(0, 1, len(fusions)))
def draw(data, save=False):
    nrows = len(replace_types)
    data_keys = list(data.keys())
    plt.figure(figsize=(2 * 10, nrows * 10))
    for i, replace in enumerate(replace_types):
        plt.subplot(nrows, 2, i * 2 + 1)

        to_break = True
        for name, xs in data.items():
            if replace in xs:
                to_break = False
                plt.plot(xs[replace] if replace != "single" else sorted(xs[replace], reverse=True), label=name.lower().replace('_', '-'))
        if to_break:
            continue

        plt.legend()
        plt.xlabel("Number of changed layer" + ("s" if replace == "seq" else ""))
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy after changing {replace} layer (from best to worst)")

        plt.subplot(nrows, 2, i * 2 + 2)


        vp = plt.violinplot([data[key][replace] for key in data_keys if replace in data[key]])
        for i, body in enumerate(vp['bodies']):
            body.set_facecolor(colors[i])
        plt.legend(list(map(lambda name:name.lower().replace('_', '-'), data_keys)))
        # plt.ylim((0, 1.1))
        # plt.xlabel("Number of changed layer" + ("s" if replace == "seq" else ""))
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy after changing {replace} layer")

    if save:
        os.makedirs(f"plots/accuracy", exist_ok=True)
        plt.savefig(f"plots/accuracy/accuracy_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.png")
    plt.show()
    plt.close()


loading = "\n".join([
    f"replace_with_{replace}_{fusion} = torch.load('{ACCURACY_DIR}/accuracy_list_{replace}_{fusion}.pt')"
    for fusion in fusions
    for replace in replace_types
    if os.path.exists(f"{ACCURACY_DIR}/accuracy_list_{replace}_{fusion}.pt")
])
exec(loading)

def_data = "\n".join(
    ["data = {"] + [
    f"   'Replace with {fusion.replace("Projection", "Pruning")}': {'{' + ', '.join([f'{"'" + replace + "'"}: replace_with_{replace}_{fusion}' for replace in replace_types if os.path.exists(f"{ACCURACY_DIR}/accuracy_list_{replace}_{fusion}.pt")]) + '}'},"
        for fusion in fusions
    ] + ["}"]
)

print(def_data)
exec(def_data)

for key1 in data.keys():
    for key2 in data[key1].keys():
        print(key1, key2, data[key1][key2])
        print()


draw(data, save=True)


data = {
   # 'Replace with ZipIt': {'seq': replace_with_seq_ZipIt, 'single': replace_with_single_ZipIt},
   # 'Replace with Average': {'seq': replace_with_seq_Average, 'single': replace_with_single_Average},
   # 'Replace with Pruning': {'seq': replace_with_seq_Projection, 'single': replace_with_single_Projection},
   # 'Replace with Procrustes': {'seq': replace_with_seq_Procrustes, 'single': replace_with_single_Procrustes},
   # 'Replace with Fisher': {'seq': replace_with_seq_Fisher, 'single': replace_with_single_Fisher},
   # 'Replace with TIES': {'seq': replace_with_seq_TIES, 'single': replace_with_single_TIES},
   # 'Replace with Distillation': {'seq': replace_with_seq_Distillation, 'single': replace_with_single_Distillation},
   # 'Replace with FineTune_ZipIt': {'seq': replace_with_seq_FineTune_ZipIt, 'single': replace_with_single_FineTune_ZipIt},
   # 'Replace with FineTune_Average': {'seq': replace_with_seq_FineTune_Average, 'single': replace_with_single_FineTune_Average},
   # 'Replace with FineTune_Pruning': {'seq': replace_with_seq_FineTune_Projection, 'single': replace_with_single_FineTune_Projection},
   # 'Replace with FineTune_Procrustes': {'seq': replace_with_seq_FineTune_Procrustes, 'single': replace_with_single_FineTune_Procrustes},
   # 'Replace with FineTune_Fisher': {'seq': replace_with_seq_FineTune_Fisher, 'single': replace_with_single_FineTune_Fisher},
   # 'Replace with FineTune_TIES': {'seq': replace_with_seq_FineTune_TIES, 'single': replace_with_single_FineTune_TIES},
   # 'Replace with FineTune_Distillation': {'seq': replace_with_seq_FineTune_Distillation, 'single': replace_with_single_FineTune_Distillation},
   # 'Replace with zero': {'seq': replace_with_seq_zero, 'single': replace_with_single_zero},
   # 'Replace with Best_MSE': {'seq': replace_with_seq_Best_MSE, 'single': replace_with_single_Best_MSE},
   # 'Replace with Best_RMSE': {'seq': replace_with_seq_Best_RMSE, 'single': replace_with_single_Best_RMSE},
   # 'Replace with Best_MAE': {'seq': replace_with_seq_Best_MAE, 'single': replace_with_single_Best_MAE},
   # 'Replace with Best_RelMSE': {'seq': replace_with_seq_Best_RelMSE, 'single': replace_with_single_Best_RelMSE},
   # 'Replace with Best_Cosine': {'seq': replace_with_seq_Best_Cosine, 'single': replace_with_single_Best_Cosine},
   # 'Replace with Best_Pearson': {'seq': replace_with_seq_Best_Pearson, 'single': replace_with_single_Best_Pearson},
   # 'Replace with Best_R2': {'seq': replace_with_seq_Best_R2, 'single': replace_with_single_Best_R2},
   # 'Replace with Best_JS': {'seq': replace_with_seq_Best_JS, 'single': replace_with_single_Best_JS},
   # 'Replace with Best_KL': {'seq': replace_with_seq_Best_KL, 'single': replace_with_single_Best_KL},
   # 'Replace with Best_Top1': {'seq': replace_with_seq_Best_Top1, 'single': replace_with_single_Best_Top1},
   # 'Replace with Best_Top5': {'seq': replace_with_seq_Best_Top5, 'single': replace_with_single_Best_Top5},
   # 'Replace with Best_EntropyGap': {'seq': replace_with_seq_Best_EntropyGap, 'single': replace_with_single_Best_EntropyGap},
   # 'Replace with Best_CKA': {'seq': replace_with_seq_Best_CKA, 'single': replace_with_single_Best_CKA},
}

draw(data)


exec(def_data)


import pandas as pd
from openpyxl import load_workbook
from openpyxl.formatting.rule import ColorScaleRule

tables_path = os.path.join(ACCURACY_DIR, 'accuracy_tables.xlsx')
with pd.ExcelWriter(tables_path, engine='openpyxl') as writer:
    for replace in replace_types:
        rows = {}
        for method, result in data.items():
            if replace not in result:
                continue

            values = list(result[replace])
            if replace == 'single':
                values = sorted(values, reverse=True)

            rows[method] = values

        if not rows:
            continue

        max_len = max(len(values) for values in rows.values())
        df = pd.DataFrame(
            {method: values + [np.nan] * (max_len - len(values)) for method, values in rows.items()},
            index=range(1, max_len + 1),
        ).T
        df.columns = list(range(1, max_len + 1))
        df.index.name = 'Method'
        df.to_excel(writer, sheet_name=replace)

workbook = load_workbook(tables_path)
for sheet_name in workbook.sheetnames:
    ws = workbook[sheet_name]
    if ws.max_row > 1 and ws.max_column > 1:
        ws.conditional_formatting.add(
            f'B2:{ws.cell(ws.max_row, ws.max_column).coordinate}',
            ColorScaleRule(
                start_type='min', start_color='FFFFFF',
                mid_type='percentile', mid_value=50, mid_color='FFF2CC',
                end_type='max', end_color='63BE7B',
            ),
        )

workbook.save(tables_path)
print(f'Tables saved to {tables_path}')


nrows = len(replace_types)
plt.figure(figsize=(2 * 10, nrows * 10))
data_keys = list(data.keys())
for ind, replace in enumerate(replace_types):
    zipped = list(zip(*[data[key][replace] for key in data_keys]))
    ordered = [[sum([zipped[i][k] < zipped[i][j] for k in range(len(zipped[0]))])
    for i in range(len(zipped))] for j in range(len(zipped[0]))]

    plt.axis(False)
    plt.subplot(nrows, 2, 2*ind + 1)
    plt.title(f"Rank for {replace} (higher is better)")
    for i, key in enumerate(data_keys):
        plt.plot(ordered[i], label = key)
    plt.legend()

    plt.subplot(nrows, 2, 2*ind + 2)
    plt.title(f"Rank for {replace} (higher is better)")
    vp = plt.violinplot(ordered)
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(colors[i])
    plt.legend(data_keys)

    for ind, key in enumerate(data_keys):
        print(f"{sum(ordered[ind]) / len(ordered[ind]):.2f} - Mean rank for {replace} {key}")
if True:
    os.makedirs(f"plots/accuracy", exist_ok=True)
    plt.savefig(f"plots/accuracy/rank_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.png")
plt.show()
plt.close()


logs = torch.load(f"logs/fusion/{list(sorted(list(os.walk("logs/fusion"))[0][2]))[-1]}")
logs = [{k:v for k, v in log.items() if k[:4] != "Best"} for log in logs]


len(logs)


plt.figure(figsize=(len(best_metrics) * 7, 7))
for ind, (metric, inverse) in enumerate(best_metrics.items()):
    plt.subplot(1, len(best_metrics), ind+1)
    def get_best(results):
        best_key = list(results.keys())[0]
        for key in results.keys():
            if (results[key][metric] < results[best_key][metric]) != inverse:
                best_key = key
        return best_key
    best = [get_best(fusion) for fusion in logs]
    plt.hist(best)
    plt.title(f"Best {metric}")
plt.show()


plt.plot([- result["ZipIt"]["Cosine"] + result["FineTune_Projection"]["Cosine"] for result in logs])
plt.ylim(-1, 1)


plt.plot(sorted([result["FineTune_Fisher"]["Cosine"] for result in logs]))
plt.plot(sorted([result["Projection"]["Cosine"] for result in logs]))
plt.ylim(-1, 1)


lens = []
for ind in range(24):
    lens.extend(list(map(len, torch.load(f"data/activations/pairs_train_{ind}.pt"))))
lens = sorted(lens)

