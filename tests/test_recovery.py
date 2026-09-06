import numpy as np
from navieco.recovery import fixed_w_kl_update, generalized_kl_divergence, recover_h


def test_fixed_w_update_preserves_nonnegativity():
    W = np.array([[1.0,0.1],[0.8,0.2],[0.1,1.0],[0.2,0.8]])
    V = np.array([[1.0,0.2],[0.8,0.1],[0.1,1.0],[0.2,0.8]])
    H = fixed_w_kl_update(V, W, np.full((2,2), 0.5))
    assert H.shape == (2,2)
    assert np.all(H >= 0)
    assert np.all(np.isfinite(H))


def test_kl_decreases_after_update():
    W = np.array([[1.0,0.1],[0.8,0.2],[0.1,1.0],[0.2,0.8]])
    H_true = np.array([[1.0,0.1],[0.1,1.0]])
    V = W @ H_true
    H0 = np.full((2,2), 0.5)
    before = generalized_kl_divergence(V, W @ H0)
    H1 = fixed_w_kl_update(V, W, H0)
    after = generalized_kl_divergence(V, W @ H1)
    assert after <= before


def test_recover_h_recovers_dominant_state():
    W = np.array([[1.0,0.05],[0.8,0.10],[0.05,1.0],[0.10,0.8]])
    H_true = np.array([[1.0,0.05,0.8],[0.05,1.0,0.2]])
    V = W @ H_true
    H = recover_h(V, W, max_iter=1000, tol=1e-8)
    assert np.array_equal(np.argmax(H, axis=0), np.argmax(H_true, axis=0))


def test_recover_parent_end_to_end(tmp_path):
    import json
    import pandas as pd
    from navieco.reference import ParentReference
    from navieco.recovery import recover_parent

    p=tmp_path/"Treg"; p.mkdir()
    genes=["A","B"]
    W=np.array([[1.0,0.05],[0.05,1.0],[0.05,1.0],[1.0,0.05]],dtype=float)
    np.save(p/"W.npy",W)
    (p/"genes.txt").write_text("A\nB\n")
    (p/"initial_states.txt").write_text("IS01\nIS02\n")
    pd.DataFrame({"initial_state":["IS01","IS02"],"final_state":["S01",np.nan],"retained":[True,False]}).to_csv(p/"state_mapping.csv",index=False)
    pd.DataFrame({"initial_state":["IS01","IS02"],"marker_count":[20,5],"AFI":[0.2,1.2]}).to_csv(p/"state_qc.csv",index=False)
    (p/"metadata.json").write_text(json.dumps({"parent":"Treg"}))

    ref=ParentReference.from_dir(p)
    X=np.array([[90,10],[10,90],[80,20],[20,80]],dtype=float)
    result,H=recover_parent(X,["A","B"],ref,input_scale="counts",cell_names=["c1","c2","c3","c4"])

    assert H.shape==(2,4)
    assert np.allclose(H.sum(axis=0),1)
    assert result.loc["c1","InitialState"]=="IS01"
    assert result.loc["c1","State"]=="S01"
    assert result.loc["c2","InitialState"]=="IS02"
    assert result.loc["c2","State"]=="Unassigned"
    assert result.loc["c2","UnassignedReason"]=="Filtered_initial_state"


def test_recover_states_partial_parents(tmp_path):
    import json
    import pandas as pd
    import anndata as ad
    from navieco.reference import NAViECOReference
    from navieco.recovery import recover_states

    root=tmp_path/"ref"; p=root/"parents"/"Treg"; p.mkdir(parents=True)
    np.save(p/"W.npy",np.array([[1,.05],[.05,1],[.05,1],[1,.05]],dtype=float))
    (p/"genes.txt").write_text("A\nB\n"); (p/"initial_states.txt").write_text("IS01\nIS02\n")
    pd.DataFrame({"initial_state":["IS01","IS02"],"final_state":["S01",np.nan],"retained":[True,False]}).to_csv(p/"state_mapping.csv",index=False)
    pd.DataFrame({"initial_state":["IS01","IS02"],"marker_count":[20,5],"AFI":[.2,1.2]}).to_csv(p/"state_qc.csv",index=False)
    (p/"metadata.json").write_text(json.dumps({"parent":"Treg"}))
    (root/"manifest.json").write_text(json.dumps({"name":"test","version":"test","layer":1,"parents":["Treg"]}))

    X=np.array([[90,10],[10,90],[80,20],[20,80],[5,5]],dtype=float)
    a=ad.AnnData(X=X)
    a.var_names=["A","B"]; a.obs_names=["c1","c2","c3","c4","c5"]
    a.layers["counts"]=X.copy()
    a.obs["NMF_parent"]=["Treg","Treg","Treg","Treg","Macrophage"]

    ref=NAViECOReference.from_dir(root)
    result=recover_states(a,ref,parent_key="NMF_parent",input_layer="counts",input_scale="counts")

    assert result.index.tolist()==a.obs_names.tolist()
    assert result.loc["c1","State"]=="S01"
    assert result.loc["c2","State"]=="Unassigned"
    assert result.loc["c2","UnassignedReason"]=="Filtered_initial_state"
    assert result.loc["c5","State"]=="Unassigned"
    assert result.loc["c5","UnassignedReason"]=="Unsupported_parent"
