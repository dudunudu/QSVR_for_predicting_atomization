#!/usr/bin/env python3
import numpy as np
import scipy.io
from scipy.linalg import eigh
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler

# 1) Loads QM7, extract 23 eigenvalues per Coulomb matrix => 23 features
def load_qm7_23d(matfile='qm7.mat'):
    data = scipy.io.loadmat(matfile)
    C_matrices = data['X']      # shape (N,23,23)
    energies   = data['T'].ravel()
    X = []
    for M in C_matrices:
        M = 0.5*(M + M.T)
        e = eigh(M, eigvals_only=True)
        X.append(np.sort(e))
    return np.array(X), energies

# 2) Feature scaling to [0,1]
def scale_features(X, feature_range=(0,1)):
    a, b = feature_range
    Xs = np.zeros_like(X)
    for j in range(X.shape[1]):
        col = X[:,j]
        mn, mx = col.min(), col.max()
        if mx > mn:
            Xs[:,j] = a + (col-mn)*(b-a)/(mx-mn)
        else:
            Xs[:,j] = col
    return Xs

# 3) ZZ‐feature map circuit
def zz_feature_map_circuit(x, reps=1):
    dim = len(x)
    qc = QuantumCircuit(dim)
    for q in range(dim):
        qc.h(q)
    for _ in range(reps):
        for i in range(dim):
            qc.rz(2*x[i], i)
        for i in range(dim):
            for j in range(i+1, dim):
                qc.cx(i,j)
                qc.rz(2*x[i]*x[j], j)
                qc.cx(i,j)
    return qc

# 4) SWAP‐test on two ZZ states
def swap_test_zz_circuit(x_i, x_j, reps=1):
    dim = 23
    total = 1 + 2*dim
    qc = QuantumCircuit(total, 1)
    anc   = 0
    reg_i = list(range(1,1+dim))
    reg_j = list(range(1+dim,1+2*dim))
    qc.compose(zz_feature_map_circuit(x_i, reps), qubits=reg_i, inplace=True)
    qc.compose(zz_feature_map_circuit(x_j, reps), qubits=reg_j, inplace=True)
    qc.h(anc)
    for k in range(dim):
        qc.cswap(anc, reg_i[k], reg_j[k])
    qc.h(anc)
    qc.measure(anc, 0)
    return qc

# 5) Builds all pairwise SWAP‐test circuits (intra‐set)
def build_swaptest_zz_circuits(X, reps=1):
    circuits, pairs = [], []
    N = len(X)
    for i in range(N):
        for j in range(i, N):
            circuits.append(swap_test_zz_circuit(X[i], X[j], reps))
            pairs.append((i,j))
    return circuits, pairs

# 6a) Compute intra‐set kernel
def compute_zz_kernel_hardware(X, sampler, backend, reps=1, shots=1024):
    circuits, pairs = build_swaptest_zz_circuits(X, reps)
    print(f"Transpiling {len(circuits)} circuits…")
    tcircs = transpile(circuits, backend=backend, optimization_level=3)
    print("Running sampler…")
    result = sampler.run(tcircs, shots=shots).result()
    N = len(X)
    K = np.zeros((N,N))
    for idx, (i,j) in enumerate(pairs):
        counts = result[idx].data.c.get_counts()
        p0 = counts.get("0",0)/shots
        fid = max(0.0, 2*p0 - 1)
        K[i,j] = K[j,i] = fid
    return K

# 6b) Computes cross‐kernel (test × train)
def compute_zz_cross_kernel_hardware(X_test, X_train, sampler, backend, reps=1, shots=1024):
    circuits, pairs = [], []
    Nt, Nn = len(X_test), len(X_train)
    for i in range(Nt):
        for j in range(Nn):
            circuits.append(swap_test_zz_circuit(X_test[i], X_train[j], reps))
            pairs.append((i,j))
    print(f"Transpiling {len(circuits)} cross-kernel circuits…")
    tcircs = transpile(circuits, backend=backend, optimization_level=3)
    print("Running sampler on cross-circuits…")
    result = sampler.run(tcircs, shots=shots).result()
    K = np.zeros((Nt, Nn))
    for idx, (i,j) in enumerate(pairs):
        counts = result[idx].data.c.get_counts()
        p0 = counts.get("0",0)/shots
        fid = max(0.0, 2*p0 - 1)
        K[i,j] = fid
    return K

# 7) Main: run QSVR on hardware
def main_zz_swaptest_hardware(
    ibm_token, backend_name, matfile,
    subset_size, reps=1, shots=1024,
    test_size=0.2, random_seed=42
):
    # loads and scale
    X, Y = load_qm7_23d(matfile)
    X = scale_features(X,(0,1))

    # subsamples & split
    np.random.seed(random_seed)
    idx = np.random.choice(len(X), subset_size, False)
    X_sub, Y_sub = X[idx], Y[idx]
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, Y_sub, test_size=test_size, random_state=random_seed
    )

    # connects & choose backend
    print("\nLogging into IBM Quantum via QiskitRuntimeService…")
    service = QiskitRuntimeService(token=ibm_token, channel="ibm_quantum")
    available = service.backends()
    print("Available backends:")
    for b in available: print(" ", b.name)
    matches = [b for b in available if b.name == backend_name]
    if not matches:
        raise ValueError(f"No such backend '{backend_name}'.")
    backend = matches[0]
    print(f"Using backend {backend.name} ({backend.configuration().num_qubits} qubits)\n")

    with Session(backend=backend) as sess:
        sampler = Sampler(sess)
        # train‐kernel
        print("=== Training kernel ===")
        K_train = compute_zz_kernel_hardware(X_train, sampler, backend, reps, shots)
        svr = SVR(kernel="precomputed", C=10.0, epsilon=1.0)
        svr.fit(K_train, y_train)

        # cross‐kernel for test
        print("=== Cross-kernel (test vs train) ===")
        K_test = compute_zz_cross_kernel_hardware(
            X_test, X_train, sampler, backend, reps, shots
        )

    # predicts & evaluate
    y_pred = svr.predict(K_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R^2 :", r2_score(y_test, y_pred))

if __name__ == "__main__":
    main_zz_swaptest_hardware(
        ibm_token="PUT_YOUR_TOKE_HERE",
        backend_name="ibm_brisbane",
        matfile="qm7.mat",
        subset_size=10,
        reps=1,
        shots=1024,
        test_size=0.2,
        random_seed=42
    )


