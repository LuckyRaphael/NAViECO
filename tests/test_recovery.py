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
