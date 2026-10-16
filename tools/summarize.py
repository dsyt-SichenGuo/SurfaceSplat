"""Collect per-scene results of an output root into a table (and results.csv).

    python tools/summarize.py output/dtu
"""
import csv
import json
import os
import sys


def main(out_root):
    rows = []
    for scene in sorted(os.listdir(out_root)):
        mesh_res = os.path.join(out_root, scene, 'fine', 'meshes', 'results.json')
        nvs_res = os.path.join(out_root, scene, 'gs', 'results.json')
        if not os.path.isfile(mesh_res) and not os.path.isfile(nvs_res):
            continue
        row = {'scene': scene}
        if os.path.isfile(mesh_res):
            row.update(json.load(open(mesh_res)))
        if os.path.isfile(nvs_res):
            nvs = json.load(open(nvs_res))
            for k, v in next(iter(nvs.values())).items():
                if k != 'num_views':
                    row[k] = v
        rows.append(row)
    if not rows:
        print('no results found in', out_root)
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    mean = {'scene': 'mean'}
    for k in keys[1:]:
        vals = [r[k] for r in rows if k in r]
        mean[k] = sum(vals) / len(vals)
    rows.append(mean)
    with open(os.path.join(out_root, 'results.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(' '.join(f'{k:>14}' for k in keys))
    for r in rows:
        print(' '.join(f'{r[k]:>14.4f}' if isinstance(r.get(k), float) else f'{str(r.get(k, "-")):>14}' for k in keys))


if __name__ == '__main__':
    main(sys.argv[1])
