import os
from pathlib import Path
import json
import sys

sys.path.insert(0, os.path.join(str(Path.home()),
                                'iFSGM/few-shot-generative-models'))



if __name__ == "__main__":
    root = "/home/gigi/ns_output/omniglot_ns_trts/log"

    data = {}
    for model in ["HFSGM", "NS", "TNS"]:
        list_days = os.listdir(os.path.join(root, model))
        for day in list_days:
            list_runs = os.listdir(os.path.join(root, model, day))
            for run in list_runs:
                name = model + "_" + day + "-" + run.split("_")[0] + "_400.json"
                print(name)
                
                path = os.path.join(root, model, day, run, name)
                try:
                    with open(path, "r") as f:
                        data[run] = json.load(f)
                        print(data[name]["test"].keys())
                except:
                    continue
    print(
        "{0:<30s} & {1:<5s} & {2:<5s} & {3:<5s} & {4:<5s} & {5:<5s} & {6:<5s} \\\\".format(
            "name",
            "model",
            "loss",
            "vlb",
            "logpx",
            "kl_z",
            "kl_c",
        )
    )

    for k in data:
        print(
            "{0:<30s} & {1:<5s} & {2:<.2f} & {3:<.2f} & {4:<.2f} & {5:<.2f} & {6:<.2f}\\\\".format(
                k,
                k.split("_")[1],
                float(data[k]["test"]["loss"][-1]),
                float(data[k]["test"]["vlb"][-1]),
                float(data[k]["test"]["logpx"][-1]),
                float(data[k]["test"]["kl_z"][-1]),
                float(data[k]["test"]["kl_c"][-1]),
            )
        )
