# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
"""Build a local HTML/JSON index of immutable Gluon MoE experiments."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from .gluon_moe_gemm1 import read_single_tg_stats, write_json


def host_path(path):
    return str(path).replace("/zhenxin/", "/home/zhenx/", 1)


def load_experiment(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("gemm") not in ("gemm1", "gemm2"):
        return None
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    expected = set(manifest.get("modes", ["non_wasp", "wasp"]))
    if "--include-wdra" in manifest.get("command", []):
        expected.add("wdra")
    status = ("failed" if any(v.get("status") == "failed" for v in summary.values()) else
              "passed" if expected <= summary.keys() and all(v.get("status") == "passed"
                                                             for v in summary.values()) else "incomplete")
    errors, modes, input_hashes = [], {}, []
    for name, result in summary.items():
        validation_path = directory / name / "validation.json"
        validation = json.loads(validation_path.read_text()) if validation_path.is_file() else None
        modes[name] = {
            "status": result.get("status"), "timing": result.get("timing"), "validation": validation, "kernel_hash":
            result.get("kernel_hash")
        }
        if result.get("status") != "passed":
            continue
        input_hashes.append(result.get("route", {}).get("input_sha256"))
        try:
            timing = read_single_tg_stats(directory / name)
            stored = result["timing"]
            if (timing["sha256"] != stored["sha256"]
                    or not math.isclose(timing["time_ns"], stored["time_ns"], rel_tol=1e-12)):
                errors.append(f"{name}: stored timing does not match raw stats")
        except (KeyError, ValueError, OSError) as error:
            errors.append(f"{name}: {error}")
        if validation is None:
            errors.append(f"{name}: numerical validation is missing")
        elif (validation["checked_elements"] <= 0
              or validation["mismatched_elements"] or validation["nonfinite_elements"]):
            errors.append(f"{name}: numerical validation failed")
    if input_hashes and (input_hashes[0] is None or any(value != input_hashes[0] for value in input_hashes)):
        errors.append("paired input hashes differ or are missing")
    baseline = manifest.get("triton_baseline", {})
    selection = baseline.get("json_config")
    if selection and manifest["recipe"] != "fp8_channel":
        errors.append("FP8 channel JSON cannot qualify a different recipe")
    if selection and "triton" in summary and summary["triton"].get("status") == "passed":
        try:
            snapshot = directory / "triton-baseline.json"
            if hashlib.sha256(snapshot.read_bytes()).hexdigest() != selection["sha256"]:
                errors.append("Triton baseline JSON snapshot hash differs")
            selected = json.loads(snapshot.read_text())[str(selection["selected_M"])]
            launch = json.loads((directory / "triton/launch.json").read_text())["options"]
            metadata = json.loads((directory / "triton/compiled.json").read_text())["kernel_metadata"]
            if selected != selection["options"] or selected != baseline["options"]:
                errors.append("Triton baseline settings differ from the selected JSON entry")
            for key, value in selected.items():
                if launch.get(key) != value or (key in metadata and metadata[key] != value):
                    errors.append(f"Triton baseline JSON setting did not take effect: {key}")
            for key, parameter in (("bm", "BLOCK_SIZE_M"), ("bn", "BLOCK_SIZE_N")):
                if manifest["config"][key] != selected[parameter]:
                    errors.append(f"paired output tiles differ: {parameter}")
        except (KeyError, ValueError, OSError) as error:
            errors.append(f"Triton baseline provenance: {error}")
    if status == "passed" and errors:
        status = "invalid"
    source = {
        name: {key: manifest.get(name, {}).get(key)
               for key in ("commit", "diff_sha256", "changed_source_sha256")}
        for name in ("triton", "boltops")
    }
    version = {
        "sources": source, "libtriton": manifest.get("libtriton"), "toolchain": manifest.get("toolchain"), "pmd":
        manifest.get("pmd")
    }
    source_id = hashlib.sha256(json.dumps(version, sort_keys=True).encode()).hexdigest()[:12]
    speedups = {}
    if status == "passed" and "non_wasp" in modes:
        baseline = modes["non_wasp"]["timing"]["time_ns"]
        speedups = {name: baseline / value["timing"]["time_ns"] for name, value in modes.items()}
    triton_speedups = {}
    if status == "passed" and "triton" in modes:
        baseline = modes["triton"]["timing"]["time_ns"]
        triton_speedups = {name: baseline / value["timing"]["time_ns"] for name, value in modes.items()}
    absolute_errors = [
        value["validation"]["max_absolute_error"]
        for value in modes.values()
        if value["validation"] and value["validation"].get("max_absolute_error") is not None
    ]
    return {
        "id": directory.name, "path": host_path(directory), "gemm": manifest["gemm"], "case": manifest["case"],
        "recipe": manifest["recipe"], "config": manifest["config"], "status": status, "integrity_errors": errors,
        "source_id": source_id, "versions": version, "valid_rows": next(
            (v.get("route", {}).get("valid_rows_in_tg") for v in summary.values() if v.get("route")),
            None), "modes": modes, "speedups": speedups, "triton_baseline": manifest.get("triton_baseline"),
        "speedups_vs_triton": triton_speedups, "max_absolute_error": max(absolute_errors) if absolute_errors else None
    }


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Gluon MoE 实验索引</title>
<style>:root{color-scheme:light}*{box-sizing:border-box}body{max-width:1200px;margin:auto;padding:30px 24px 80px;background:#fff;color:#252525;font:14px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC",Arial,sans-serif}h1{font-size:26px;border-bottom:2px solid #e1e4e8}a{color:#0366d6}code{background:#eef2f6;padding:2px 5px}label{display:inline-block;margin:6px 18px 6px 0}select{padding:6px;border:1px solid #cbd5e1;background:#fff;color:#252525;max-width:250px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #e1e4e8;padding:7px 9px;text-align:left;vertical-align:top;overflow-wrap:anywhere}th,tr:nth-child(even){background:#f6f8fa}small{color:#555}.scroll{overflow-x:auto}td:first-child{min-width:200px;max-width:320px}td:nth-child(2){min-width:140px}.failed,.invalid{color:#a61c1c}details{margin-top:5px}#count{color:#555}</style></head><body>
<h1>Gluon MoE 实验索引</h1>
<p>单位 µs。每行只比较同一实验的普通 Triton（如有）、Gluon 非 WASP、WASP、WDRA；源码、shape、tile 或 BM/BN 不同的行不构成直接性能排名。点击记录可回到源码快照、数值结果及原始 stats。WASP 倍率低于 1 表示回退。普通 Triton 配置单独列出，Gluon/Triton 倍率使用本行最快 Gluon。</p>
<p><a href="index.json">完整 JSON</a> · <a href="/home/zhenx/obsidian-book/raw/ai-compiler/2026.11/20261102-gluon-hcu-ttgir-moe.html">设计与验收范围</a></p>
<div id="filters"></div><p id="count"></p><div class="scroll"><table><thead><tr><th>实验 / 源码版本</th><th>配置 / 有效行</th><th>普通 Triton</th><th>Gluon 非 WASP</th><th>WASP</th><th>WDRA</th><th>WASP 倍率</th><th>Gluon/Triton 倍率</th><th>最大绝对误差</th><th>记录</th></tr></thead><tbody id="rows"></tbody></table></div>
<h2>独立激活与归约</h2><p>此处独立探针不代表完整两阶段 MoE 已通过。FP8 重数量化的 MODE2 未实现见 <a href="/home/zhenx/obsidian-book/wiki/ai-compiler/pmd-known-issues.md">PMD-006</a>。</p><ul id="auxiliary"></ul>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent), entries=data.experiments, filters={};
const definitions=[['gemm','投影',e=>e.gemm],['case','case',e=>e.case.id],['recipe','配方',e=>e.recipe],['source','源码',e=>e.source_id],['status','状态',e=>e.status]];
for(const [key,title,get] of definitions){const label=document.createElement('label');label.append(title+' ');const select=document.createElement('select');select.add(new Option('全部',''));for(const value of [...new Set(entries.map(get))].sort())select.add(new Option(value,value));select.onchange=render;label.append(select);document.getElementById('filters').append(label);filters[key]=select;}
const number=x=>x==null?'—':Number(x).toFixed(3);
function anchor(path,title){const a=document.createElement('a');a.href=path;a.textContent=title;return a;}
function render(){const shown=entries.filter(e=>definitions.every(([key,,get])=>!filters[key].value||get(e)===filters[key].value));document.getElementById('count').textContent=shown.length+' / '+entries.length+' 笔实验';const body=document.getElementById('rows');body.replaceChildren();for(const e of shown){const tr=document.createElement('tr');tr.className=e.status;const c=e.config, times=['triton','non_wasp','wasp','wdra'].map(mode=>e.modes[mode]?.timing?.time_ns/1000);const values=[e.gemm+' · '+e.case.id+' · '+e.recipe+'\n'+e.id+'\n版本 '+e.source_id+' · '+e.status,[c.bm,c.bn,c.bk].join('/')+' · S'+c.stages+' · '+(c.waves??4)+' waves\nW'+(c.mma_warps??(c.bm===16?[1,4]:[2,2])).join('×')+' · tile '+c.tile+' · 有效行 '+(e.valid_rows??'—')+(c.early_release?' · 提前释放':'')+(c.prefetch?' · 显式预取':'')+(c.direct_lds?' · LDS→Dot':'')+(c.buffer_copy?' · buffer copy':'')+(c.lds_wrap?' · wrap '+c.lds_wrap:'')+(c.lds_swizzle>1?' · swizzle '+c.lds_swizzle:'')+(c.channel_n_split?' · 2P×'+c.producer_waves+' waves / 2C×'+c.waves+' waves':'')+(e.modes.triton?'\nTriton '+[e.triton_baseline.bm??c.bm,e.triton_baseline.bn??c.bn,e.triton_baseline.bk??c.bk].join('/')+(e.triton_baseline.configuration_kind==='json'?' · JSON M='+e.triton_baseline.json_config.selected_M:' · 手动基线')+' · S'+e.triton_baseline.stages+' · '+e.triton_baseline.waves+' waves'+(e.triton_baseline.async_copy?' · async':'')+(e.triton_baseline.matrix_load_b?' · MLS B':'')+(e.triton_baseline.block_pingpong?' · pingpong':''):''),...times.map(x=>Number.isFinite(x)?number(x):'—'),number(e.speedups.wasp),number(Math.max(...Object.entries(e.speedups_vs_triton??{}).filter(([m])=>m!=='triton').map(([,v])=>v),0)||null),e.max_absolute_error??'—'];for(const value of values){const td=document.createElement('td');td.textContent=value;td.style.whiteSpace='pre-line';tr.append(td);}const td=document.createElement('td');td.append(anchor(e.path+'/manifest.json','manifest'),document.createElement('br'),anchor(e.path+'/summary.json','summary'));const detail=document.createElement('details'),title=document.createElement('summary');title.textContent='源码 / stats';detail.append(title);if(e.triton_baseline?.json_config){detail.append(anchor(e.path+'/triton-baseline.json','原始基线 JSON'),document.createElement('br'));const note=document.createElement('p');note.textContent=e.triton_baseline.json_config.recipe_scope;detail.append(note);}for(const name of ['triton','boltops']){detail.append(anchor(e.path+'/sources/'+name+'/tracked.diff',name+' diff'),document.createElement('br'));}for(const [name,value] of Object.entries(e.modes)){if(value.timing)detail.append(anchor(e.path+'/'+name+'/'+value.timing.stats_path,name+' stats'),document.createElement('br'));}if(e.integrity_errors.length){const text=document.createElement('p');text.textContent=e.integrity_errors.join('; ');detail.append(text);}td.append(detail);tr.append(td);body.append(tr);}}
render();for(const e of data.auxiliary){const li=document.createElement('li');li.append(anchor(e.path+'/manifest.json',e.id),' — '+e.status);document.getElementById('auxiliary').append(li);}
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/zhenxin/temp"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    experiments, auxiliary = [], []
    for directory in sorted(args.root.iterdir()):
        if not directory.is_dir() or not (directory / "manifest.json").is_file():
            continue
        manifest = json.loads((directory / "manifest.json").read_text())
        record = load_experiment(directory)
        if record:
            experiments.append(record)
        elif "case" in manifest and manifest.get("operation") in ("activation", "reduce"):
            result_path = directory / "result.json"
            result = json.loads(result_path.read_text()) if result_path.is_file() else {}
            auxiliary.append({
                "id": directory.name, "path": host_path(directory), "status": result.get("status", "incomplete"),
                "manifest": manifest, "result": result
            })
    args.out.mkdir(parents=True, exist_ok=False)
    data = {
        "schema_version": 1, "experiments": experiments, "auxiliary": auxiliary, "comparison_contract":
        "paired modes only; no cross-shape/tile/version or BM/BN ranking"
    }
    write_json(args.out / "index.json", data)
    encoded = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    (args.out / "index.html").write_text(HTML.replace("__DATA__", encoded))
    print(f"Indexed {len(experiments)} GEMM experiments and {len(auxiliary)} auxiliary probes into {args.out}")


if __name__ == "__main__":
    main()
