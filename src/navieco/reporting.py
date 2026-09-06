"""Recovery summaries and provenance for NAViECO."""

from __future__ import annotations
import json
from pathlib import Path
from .recovery import check_query_compatibility


def build_recovery_summary(adata, assignments, reference, parent_key="NMF_parent", input_layer="counts", input_scale="counts", scale_group_keys=None):
    """Build a JSON-serializable summary of one Layer 1 recovery run."""
    compatibility=check_query_compatibility(adata,reference,parent_key=parent_key)
    assigned=assignments["State"].ne("Unassigned")
    reasons=assignments.loc[~assigned,"UnassignedReason"].fillna("Unknown").value_counts()
    states=assignments.loc[assigned,"State"].value_counts()
    return {
        "reference_name":reference.manifest.get("name"),
        "reference_version":reference.manifest.get("version"),
        "layer":reference.manifest.get("layer",1),
        "parent_key":parent_key,
        "input_layer":input_layer,
        "input_scale":input_scale,
        "scale_group_keys":list(scale_group_keys) if scale_group_keys else None,
        "n_cells":int(adata.n_obs),
        "n_assigned":int(assigned.sum()),
        "n_unassigned":int((~assigned).sum()),
        "assignment_rate":float(assigned.mean()),
        "state_counts":{str(k):int(v) for k,v in states.items()},
        "unassigned_reason_counts":{str(k):int(v) for k,v in reasons.items()},
        "compatibility":json.loads(compatibility.to_json(orient="records"))
    }


def save_recovery_summary(summary, path):
    """Write a NAViECO recovery summary to JSON."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",encoding="utf-8") as f: json.dump(summary,f,indent=2,ensure_ascii=False)
    return path
