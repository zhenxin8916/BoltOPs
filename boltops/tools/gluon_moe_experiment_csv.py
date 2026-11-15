# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Maintain normalized Gluon GEMM1/GEMM2 experiment CSVs (default GC 1.8 GHz).

These diagnostic dispatches are not full-MoE timings. BM128 and
BN512/256/128 x 2/4/8 outputs per persistent TG give equal padded output work.
Publish a validated record with --stage top|bottom --record record.json.
"""
import argparse
import csv
import json
import math
from pathlib import Path

from boltops.tools.fused_moe_triton_tune.performance_csv import _atomic_write, csv_write_lock

DEFAULT_GC_GHZ = 1.8
CONFIG_DIR = Path(__file__).resolve().parents[1] / "fused_moe/gluon"
FIELDS = (
    'arch', 'quant_type', 'indtype', 'token', 'inter_dim', 'model_dim',
    'expert', 'topk', 'sol_type', 'sol_id', 'bm', 'bn',
    'bk', 'time_us', 'triton_commit', 'triton_changes', 'llvm_project_commit', 'llvm_changes',
)
KEY_FIELDS = (
    'arch', 'quant_type', 'indtype', 'token', 'inter_dim', 'model_dim',
    'expert', 'topk', 'sol_id', 'bm', 'bn', 'bk',
    'triton_commit', 'triton_changes', 'llvm_project_commit', 'llvm_changes',
)


def tiles_per_tg(bn):
    """BN512/256/128 -> 2/4/8 tiles, each covering 128*1024 output elements."""
    if bn not in (128, 256, 512):
        raise ValueError("BN must be 128, 256 or 512")
    return 1024 // bn


def upsert_record(stage, record, directory=CONFIG_DIR):
    """Replace a matching experiment row after numerical/frequency/work validation."""
    if stage not in ("top", "bottom"):
        raise ValueError("stage must be top or bottom")
    row = dict(record)
    for key in FIELDS:
        if key not in row:
            raise ValueError(f"missing experiment field: {key}")
    if record.get("mismatches") != 0 or not record.get("verified"):
        raise ValueError("only verified, numerically correct runs may be published")
    if int(row["bm"]) != 128 or int(row["tiles_per_tg"]) != tiles_per_tg(int(row["bn"])):
        raise ValueError("expected BM128 and BN512/256/128 x 2/4/8 tiles per TG")
    if int(row["total_tiles"]) != int(row["cu_count"]) * int(row["tiles_per_tg"]):
        raise ValueError("total_tiles must give each CU one TG with equal work")
    if int(row["output_elements_per_tg"]) != 128 * 1024:
        raise ValueError("unequal padded output work per TG")
    if row["time_scope"] != "full_dispatch" or row["work_scope"] != "bounded_equal_padded_output_work":
        raise ValueError("expected an explicitly labelled diagnostic dispatch")
    for key in ("time_us", "wave_span_time_us", "gc_ghz", "effective_gc_ghz"):
        value=float(row[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if float(row["time_us"]) < float(row["wave_span_time_us"]):
        raise ValueError("full dispatch cannot be shorter than its wave span")
    if int(record.get("q_size_n", 0)) or int(record.get("q_size_k", 0)):
        raise ValueError("this table records channel scales, not block quantization")
    if not math.isclose(float(row["gc_ghz"]), DEFAULT_GC_GHZ):
        raise ValueError("this experiment table uses GC 1.8 GHz")
    # The model rounds nominal 1.8 GHz to an integer 556-ps period.
    if abs(float(row["effective_gc_ghz"])-1000/round(1000/DEFAULT_GC_GHZ)) > 1e-5:
        raise ValueError("effective model clock does not match requested GC")
    row={key:str(value) for key,value in row.items()}
    for key in ("time_us", "wave_span_time_us", "effective_gc_ghz"):
        row[key]=f"{float(row[key]):.6f}"
    row["gc_ghz"]=str(DEFAULT_GC_GHZ)
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    suffix = "_int8" if row["quant_type"] == "int8_w8a8_channel" else ("_fp8" if stage == "top" else "")
    path=directory/f"tuned_experiment_fmoe_gluon_w8a8channel_{stage}{suffix}.csv"
    key=lambda item:tuple(item[name] for name in KEY_FIELDS)
    with csv_write_lock(path):
        rows={}
        if path.exists():
            with path.open(newline="") as stream:
                reader=csv.DictReader(stream)
                if tuple(reader.fieldnames or ()) != FIELDS:
                    raise ValueError("unexpected experiment CSV header")
                rows={key(item):item for item in reader}
        rows[key(row)]={name: row[name] for name in FIELDS}
        def write(stream):
            writer=csv.DictWriter(stream,fieldnames=FIELDS,lineterminator="\n")
            writer.writeheader();writer.writerows(sorted(rows.values(),key=lambda item: (int(item["token"]), key(item))))
        _atomic_write(path,write)
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("top","bottom"),required=True)
    parser.add_argument("--record",type=Path,required=True)
    parser.add_argument("--directory",type=Path,default=CONFIG_DIR)
    args=parser.parse_args()
    print(upsert_record(args.stage,json.loads(args.record.read_text()),args.directory))


if __name__ == "__main__":
    main()
