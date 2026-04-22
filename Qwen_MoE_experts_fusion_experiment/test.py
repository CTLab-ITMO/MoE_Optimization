#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import torch
from datetime import datetime
import os
import sys
import numpy as np
sys.path.append("..")
from common.mlp_utils import best_metrics

replace_types = ["seq", "single"]
fusions = ["Average", "ZipIt", "Projection"]
fusions += [f"FineTune_{fusion}" for fusion in fusions]
fusions += ["zero"]
fusions += [f"Best_{metric}" for metric in best_metrics.keys()]


colors = plt.cm.tab20(np.linspace(0, 1, len(fusions)))
def draw(data, save=False):
    nrows = len(replace_types)
    data_keys = list(data.keys())
    plt.figure(figsize=(2 * 10, nrows * 10))
    for i, replace in enumerate(replace_types):
        plt.subplot(nrows, 2, i * 2 + 1)

        for name, xs in data.items():
            plt.plot(sorted(xs[replace], reverse=True), label=name.lower().replace('_', '-'))
        plt.legend()
        plt.xlabel("Number of changed layer" + ("s" if replace == "seq" else ""))
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy after changing {replace} layer (from best to worst)")

        plt.subplot(nrows, 2, i * 2 + 2)


        vp = plt.violinplot([data[key][replace] for key in data_keys])
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
    f"replace_with_{replace}_{fusion} = torch.load('accuracy/accuracy_list_{replace}_{fusion}.pt')"
    for fusion in fusions
    for replace in replace_types
])
exec(loading)

def_data = "\n".join(
    ["data = {"] + [
    f"   'Replace with {fusion.replace("Projection", "Pruning")}': {'{' + ', '.join([f'{"'" + replace + "'"}: replace_with_{replace}_{fusion}' for replace in replace_types]) + '}'},"
        for fusion in fusions
    ] + ["}"]
)

print(def_data)
exec(def_data)


draw(data, save=True)


data = {
   'Replace with Average': {'seq': replace_with_seq_Average, 'single': replace_with_single_Average},
   'Replace with ZipIt': {'seq': replace_with_seq_ZipIt, 'single': replace_with_single_ZipIt},
   'Replace with Pruning': {'seq': replace_with_seq_Projection, 'single': replace_with_single_Projection},
   'Replace with FineTune_Average': {'seq': replace_with_seq_FineTune_Average, 'single': replace_with_single_FineTune_Average},
   'Replace with FineTune_ZipIt': {'seq': replace_with_seq_FineTune_ZipIt, 'single': replace_with_single_FineTune_ZipIt},
   'Replace with FineTune_Pruning': {'seq': replace_with_seq_FineTune_Projection, 'single': replace_with_single_FineTune_Projection},
   'Replace with zero': {'seq': replace_with_seq_zero, 'single': replace_with_single_zero},
   'Replace with Best_MSE': {'seq': replace_with_seq_Best_MSE, 'single': replace_with_single_Best_MSE},
   'Replace with Best_Cosine': {'seq': replace_with_seq_Best_Cosine, 'single': replace_with_single_Best_Cosine},
   'Replace with Best_JS': {'seq': replace_with_seq_Best_JS, 'single': replace_with_single_Best_JS},
   'Replace with Best_CKA': {'seq': replace_with_seq_Best_CKA, 'single': replace_with_single_Best_CKA},
}
draw(data)


data = {
   # 'Replace with Average': {'seq': replace_with_seq_Average, 'single': replace_with_single_Average},
   # 'Replace with ZipIt': {'seq': replace_with_seq_ZipIt, 'single': replace_with_single_ZipIt},
   'Replace with Pruning': {'seq': replace_with_seq_Projection, 'single': replace_with_single_Projection},
   # 'Replace with FineTune_Average': {'seq': replace_with_seq_FineTune_Average, 'single': replace_with_single_FineTune_Average},
   # 'Replace with FineTune_ZipIt': {'seq': replace_with_seq_FineTune_ZipIt, 'single': replace_with_single_FineTune_ZipIt},
   'Replace with FineTune_Pruning': {'seq': replace_with_seq_FineTune_Projection, 'single': replace_with_single_FineTune_Projection},
   # 'Replace with zero': {'seq': replace_with_seq_zero, 'single': replace_with_single_zero},
   # 'Replace with Best_MSE': {'seq': replace_with_seq_Best_MSE, 'single': replace_with_single_Best_MSE},
   # 'Replace with Best_Cosine': {'seq': replace_with_seq_Best_Cosine, 'single': replace_with_single_Best_Cosine},
   # 'Replace with Best_JS': {'seq': replace_with_seq_Best_JS, 'single': replace_with_single_Best_JS},
   # 'Replace with Best_CKA': {'seq': replace_with_seq_Best_CKA, 'single': replace_with_single_Best_CKA},
}
draw(data)


exec(def_data)


nrows = len(replace_types)
plt.figure(figsize=(2 * 10, nrows * 10))
data_keys = list(data.keys())
for ind, replace in enumerate(replace_types):
    zipped = list(zip(*[data[key][replace] for key in data_keys]))
    ordered = [[
        sum([zipped[i][k] < zipped[i][j] for k in range(len(zipped[0]))])
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


plt.plot([result["ZipIt"]["Cosine"] - result["Projection"]["Cosine"] for result in logs])
plt.ylim(-1, 1)


import matplotlib.pyplot as plt
import torch
from datetime import datetime
import os
import sys
import numpy as np
sys.path.append("..")
from common.mlp_utils import best_metrics

replace_types = ["seq", "single"]
fusions = ["Average", "ZipIt", "Projection"]
fusions += [f"FineTune_{fusion}" for fusion in fusions]
fusions += ["zero"]
fusions += [f"Best_{metric}" for metric in best_metrics.keys()]


lens = []
for ind in range(24):
    lens.extend(list(map(len, torch.load(f"data/activations/pairs_train_{ind}.pt"))))
lens = sorted(lens)


plt.plot([4096 for _ in lens])
plt.plot(lens)
plt.yscale("log")

