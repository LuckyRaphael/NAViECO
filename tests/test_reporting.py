import json
import pandas as pd
from navieco.reporting import save_recovery_summary


def test_save_recovery_summary(tmp_path):
    summary={"reference_name":"NAViECO-test","reference_version":"test","n_cells":10,"assignment_rate":0.8}
    path=save_recovery_summary(summary,tmp_path/"summary.json")
    assert path.exists()
    with open(path,encoding="utf-8") as f: loaded=json.load(f)
    assert loaded==summary
