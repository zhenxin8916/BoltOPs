# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Publish only verified BM32 FP8 rows from the archived migration evidence."""
import json
from pathlib import Path

from boltops.tools.gluon_moe_experiment_csv import upsert_record

ROOT = Path(__file__).resolve().parents[4]
DOCS = Path(__file__).resolve().parent.parent
COMPILER_NOTE = (
    " 实测clang构建标识仍为fd18eddb，二进制SHA256="
    "0ec2b806f4f940fa67310006c3c2ef3ca602a0f6823538f988b0162405ea2add；"
    "前列记录源码HEAD，实际二进制身份以此哈希为准；本次未修改编译器。"
)


def publish():
    top_dir = DOCS / "gemm1-bm32-bn256-integration-20261117"
    top = json.loads((top_dir / "performance-records.json").read_text())
    assert len(top) == 3
    paths = set()
    records = {"top": [], "bottom": []}
    for item in top:
        row = dict(item)
        row["llvm_changes"] += COMPILER_NOTE
        row["wave_span_time_us"] = row["cu0_lifetime_us"]
        records["top"].append(row)
        paths.add(str(upsert_record("top", row, ROOT / "boltops/fused_moe/gluon")))
    performance = json.loads((DOCS / "gemm2-bm32-bn512-port-20261117/performance.json").read_text())
    assert len(performance) == 3
    base = records["top"][0]
    for item in performance:
        assert item["numeric"]["verified"] and item["numeric"]["mismatches"] == 0
        gc = item["frequency_ghz"]
        row = {key: base[key] for key in (
            "arch", "quant_type", "indtype", "triton_commit", "triton_changes",
            "llvm_project_commit", "llvm_changes", "token", "inter_dim", "model_dim",
            "expert", "topk", "sol_type", "cu_count", "tiles_per_tg", "total_tiles",
            "time_scope", "work_scope", "mmac_scope")}
        row.update(sol_id="triton:gluon_moe_gemm2", bm=32, bn=512, bk=64,
                   output_elements_per_tg=32*512*2, actual_n=2048, actual_k=256,
                   gc_ghz=gc, effective_gc_ghz=1000/round(1000/gc),
                   time_us=item["time_us"], wave_span_time_us=item["wave_span_time_us"],
                   mmac_total_pct=item["cu0_total_mmac_percent"],
                   hot_loop_mmac_pct=item["cu0_hot_mmac_percent"],
                   mmac_endpoint_method=item["whole_scope"],
                   verified=True, mismatches=0)
        records["bottom"].append(row)
        paths.add(str(upsert_record("bottom", row, ROOT / "boltops/fused_moe/gluon")))
    (Path(__file__).parent / "published-records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2)+"\n")
    print("\n".join(sorted(paths)))


if __name__ == "__main__":
    publish()
