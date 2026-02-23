import os
import numpy as np
from collections import defaultdict


def read_evaluations(evaluation_path):
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    with open(evaluation_path, 'r') as f:
        lines = f.readlines()
        header = lines[0].strip().split(",")
        for line in lines[1:]:
            body = line.strip().split(",")
            method = body[0]
            for i in range(1, len(body)):
                metric_meter = header[i]
                result = metric_meter.rsplit("_", 1)
                metric = result[0]
                meter = result[1]
                value = body[i]
                results[meter][metric][method] = float(value)
        return results


def tablelize(results, scenes, metrics, best_at_min, methods, meters):
    huge = []
    leader = []
    for metric_t in metrics:
        for method_t in methods:
            method = method_t[1]
            leader.append(method)
    huge.append([""] + leader)
    for meter in meters:
        table = []
        header = []
        for i, metric_t in enumerate(metrics):
            metric = metric_t[1]
            arrow = '\\downarrow' if best_at_min[i] else '\\uparrow'
            header.extend([f"{metric}$_{'{'+meter+'}'+arrow+'$'}"] + [""] * (len(methods) - 1))
        table.append([""] + header)
        for scene in scenes:
            row = []
            for i, metric_t in enumerate(metrics):
                metric = metric_t[0]
                group = []
                for method_t in methods:
                    method = method_t[0]
                    group.append(results[scene][meter][metric][method])
                if best_at_min[i]:
                    group_bf = np.array(group).min()
                else:
                    group_bf = np.array(group).max()
                for g in group:
                    if g == group_bf:
                        left = "\\textbf{"
                        right = "}"
                        row.append(f"{left}{g:.2f}{right}")
                    else:
                        row.append(f"{g:.2f}")
            table.append([scene] + row)
        huge.append(table)

    str_rows = []
    for i in range(len(huge)):
        table = huge[i]
        if i == 0:
            str_rows.append('\\toprule')
            str_row = "\multicolumn{1}{c}{\multirow{2}{*}{Seq}}"
            for item in table[1:]:
                str_row += " & \multicolumn{1}{c}{\\textbf{" + item + "}}"
            str_rows.append(str_row + '\\\\')
            str_row = ""
            for k in range(len(metrics)):
                str_row += "\cmidrule(r){" + str(len(methods) * k + 2) + "-" + str(len(methods) * k + len(methods) + 1) + "}"
            str_rows.append(str_row)
        else:
            for j in range(len(table)):
                row = table[j]
                if j == 0:
                    str_row = "\multicolumn{1}{c}{}"
                    for item in row:
                        if item != "":
                            str_row += " & \multicolumn{" + str(len(methods)) + "}{c}{" + item + "}"
                    str_rows.append(str_row + '\\\\')
                    str_row = ""
                    for k in range(len(metrics)):
                        str_row += "\cmidrule(r){" + str(len(methods) * k + 2) + "-" + str(len(methods) * k + len(methods) + 1) + "}"
                    str_rows.append(str_row)
                else:
                    str_row = ' & '.join(row)
                    str_rows.append(str_row + '\\\\')
            str_rows.append('\midrule')
    res = '\n'.join(str_rows)
    print(res)


if __name__ == "__main__":
    seq_ids = os.getenv('DEPLOY_SEQS').split(",")# + ["all"]

    output_dir = os.getenv('OUTPUT_DIR') + ('/inference' if os.getenv('MINISET') is None else '/inference.mini')
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for seq_id in seq_ids:
        evaluations = read_evaluations(os.path.join(output_dir, f"trajectory_evaluation_{seq_id}.txt"))
        scene = seq_id
        results[scene] = evaluations

    # Fill the storage structure
    scenes = seq_ids
    # metrics = [("ADE", "ADE"), ("FDE", "FDE"), ("HitRate", "HitRate"), ("cover_interp_IoU", "CVG")]
    # best_at_min = [True, True, False, False]
    # methods = [("skeleton_scoring", "Se."), ("end2end", "En."), ("pred_consume_invariant_beta_unifroms15", "RRT."), ("pred_consume_invariant_beta_rnt_unifroms15", "Bez.")]
    # meters = ["10m", "20m"]
    metrics = [("ADE", "ADE")]
    best_at_min = [True]
    methods = [("ogm", "Mo."), ("baseline_1", "Di."), ("baseline_2", "Di.$^\prime$"), ("edt_1", "Gr."), ("edt_2", "Gr.$^\prime$"), ("simple", "Ne."), ("naive", "IOr."), ("pred_consume_invariant", "FS-DOr."), ("pred_consume_invariant_beta_unifroms15", "DOr."), ("pred_consume_invariant_beta_rnt_unifroms15", "DOr.Bez.")]
    meters = ["10m", "20m"]

    tablelize(results, scenes, metrics, best_at_min, methods, meters)
