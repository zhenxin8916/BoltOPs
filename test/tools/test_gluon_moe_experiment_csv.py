import csv
import pytest
from boltops.tools.gluon_moe_experiment_csv import FIELDS, tiles_per_tg, upsert_record


def record(bn=256):
    row={key:"sample" for key in FIELDS}
    row.update(arch="gfx946",quant_type="f8_w8a8_channel",indtype="torch.bfloat16",
               token=4096,inter_dim=512,model_dim=2048,expert=256,topk=8,q_size_n=0,q_size_k=0,
               sol_type="triton",sol_id="triton:gluon_moe_gemm1",time_us=60,
               wave_span_time_us=56,gc_ghz=1.8,effective_gc_ghz=1000/556,
               cu_count=48,bm=128,bn=bn,bk=128,tiles_per_tg=1024//bn,
               total_tiles=48*1024//bn,output_elements_per_tg=128*1024,
               actual_n=1024,actual_k=2048,mmac_scope="cu0_wave_lifetime",mmac_total_pct=70, time_scope="full_dispatch",
               work_scope="bounded_equal_padded_output_work",warmup=0,mismatches=0,verified=True)
    return row


@pytest.mark.parametrize("bn,count",[(128,8),(256,4),(512,2)])
def test_normalized_work(bn,count):
    assert tiles_per_tg(bn)==count
    assert 128*bn*count==128*1024


@pytest.mark.parametrize("change",[
    {"gc_ghz":2.0},{"effective_gc_ghz":2.197802},{"tiles_per_tg":2},
    {"total_tiles":96},{"mismatches":1},{"verified":False},
    {"time_scope":"wave_span"},{"output_elements_per_tg":65536},
    {"time_us":float("nan")},{"time_us":1},{"q_size_k":128},
])
def test_reject_incomparable_or_unverified_results(tmp_path,change):
    row=record();row.update(change)
    with pytest.raises(ValueError):upsert_record("top",row,tmp_path)
    assert not list(tmp_path.glob("*.csv"))


def test_replaces_only_matching_row_and_keeps_stages_separate(tmp_path):
    a=record(256);path=upsert_record("top",a,tmp_path)
    a["time_us"]=58;upsert_record("top",a,tmp_path)
    upsert_record("top",record(128),tmp_path)
    bottom=upsert_record("bottom",record(512),tmp_path)
    with path.open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==2
    assert next(row for row in rows if row["bn"]=="256")["time_us_1_8ghz"]=="58.000000"
    with bottom.open() as stream:assert len(list(csv.DictReader(stream)))==1


def test_frequency_updates_preserve_other_measurements(tmp_path):
    row = record()
    path = upsert_record("top", row, tmp_path)
    row.update(gc_ghz=2.2, effective_gc_ghz=1000/455, time_us=55,
               wave_span_time_us=51, mmac_total_pct=65)
    upsert_record("top", row, tmp_path)
    row.update(gc_ghz=1.5, effective_gc_ghz=1000/667, time_us=68,
               wave_span_time_us=64, mmac_total_pct=75)
    upsert_record("top", row, tmp_path)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert [rows[0][f"time_us_{gc}ghz"] for gc in ("2_2", "1_8", "1_5")] == ["55.000000", "60.000000", "68.000000"]
    assert [rows[0][f"mmac_total_pct_{gc}ghz"] for gc in ("2_2", "1_8", "1_5")] == ["65.000000%", "70.000000%", "75.000000%"]
