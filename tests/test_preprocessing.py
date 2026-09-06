import numpy as np
from scipy import sparse
from navieco.preprocessing import to_cp10k, match_reference_genes, prepare_reference_expression, scale_query, posneg


def test_cp10k_dense_and_sparse():
    X=np.array([[1,1,2],[0,2,2]],dtype=float)
    for Y in (to_cp10k(X),to_cp10k(sparse.csr_matrix(X))):
        Y=Y.toarray() if sparse.issparse(Y) else Y
        assert np.allclose(Y.sum(axis=1),1e4)


def test_reference_gene_matching_and_missing_gene():
    X=np.array([[1,2,3],[4,5,6]],dtype=float)
    Y=match_reference_genes(X,["A","B","C"],["C","A","X"])
    assert np.array_equal(Y,np.array([[3,1,0],[6,4,0]],dtype=float))


def test_prepare_reference_expression():
    X=np.array([[1,1],[3,1]],dtype=float)
    Y=prepare_reference_expression(X,["A","B"],["B","A"],input_scale="counts")
    expected=np.log2(np.array([[5000,5000],[2500,7500]],dtype=float)+1)
    assert np.allclose(Y,expected)


def test_scale_query():
    X=np.array([[1,2],[3,2],[5,8]],dtype=float)
    Z=scale_query(X)
    assert np.allclose(Z.mean(axis=0),0)
    assert np.allclose(Z.std(axis=0,ddof=1),[1,1])


def test_posneg():
    Z=np.array([[1,-2],[-3,4]],dtype=float)
    V=posneg(Z)
    assert V.shape==(4,2)
    assert np.array_equal(V,np.array([[1,0],[0,4],[0,3],[2,0]],dtype=float))
    assert np.all(V>=0)
