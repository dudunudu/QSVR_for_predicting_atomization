#!/usr/bin/env python3
import numpy as np
import scipy.io
from scipy.linalg import eigh
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler


# 1) Loads the QM7 dataset and extract 23 eigenvalues per Coulomb matrix

def load_qm7_eigenvalues(matfile="qm7.mat"):
    """
    Loads the QM7 dataset from `matfile` and compute the 23 eigenvalues
    of each 23x23 Coulomb matrix.
    
    Returns:
      X_eigs: (N, 23) array of sorted eigenvalues for each molecule
      Y: (N,) array of target energies
    """
    data = scipy.io.loadmat(matfile)
    C_matrices = data["X"]         # shape: (N, 23, 23)
    energies   = data["T"].ravel()  # shape: (N,)
    
    eigenvals = []
    for M in C_matrices:
        M = 0.5 * (M + M.T)  # ensure symmetry
        e, _ = eigh(M)
        eigenvals.append(np.sort(e))
    X_eigs = np.array(eigenvals)
    return X_eigs, energies


# 2) Amplitude encoding utility

def amplitude_encode_state(vec, num_qubits):
    """
    Builds a circuit of `num_qubits` that amplitude-encodes the 1D array `vec`.
    - Pads the vector up to length 2^num_qubits with zeros.
    - Normalizes the resulting vector.
    - Uses `initialize(...)` to set amplitudes.
    Returns a QuantumCircuit (without measurement).
    """
    dim = 2 ** num_qubits
    qc = QuantumCircuit(num_qubits)
    
    padded = np.zeros(dim, dtype=complex)
    if len(vec) > dim:
        raise ValueError(f"Vector length {len(vec)} exceeds max {dim} for {num_qubits} qubits.")
    padded[:len(vec)] = vec
    norm = np.linalg.norm(padded)
    if norm < 1e-9:
        padded[0] = 1.0
        norm = 1.0
    padded /= norm
    qc.initialize(padded, list(range(num_qubits)))
    return qc


# 3) SWAP test circuit

def swap_test_circuit(vec_i, vec_j, num_qubits=5):
    """
    Constructs a SWAP test circuit to estimate |<phi_i|phi_j>|^2 between two
    amplitude-encoded states.
    The ideal probability of measuring the ancilla in |0> is (1+F)/2,
    where F = |<phi_i|phi_j>|^2. Hence, F = 2*p0 - 1.
    """
    anc = 0
    # Registers for the two states
    reg1 = list(range(1, num_qubits + 1))
    reg2 = list(range(num_qubits + 1, 2 * num_qubits + 1))
    total_qubits = 1 + 2 * num_qubits
    
    # Create a circuit with one classical bit for the ancilla measurement.
    qc = QuantumCircuit(total_qubits, 1)
    
    # Encode the two state vectors
    qc.compose(amplitude_encode_state(vec_i, num_qubits), qubits=reg1, inplace=True)
    qc.compose(amplitude_encode_state(vec_j, num_qubits), qubits=reg2, inplace=True)
    
    # SWAP test: apply H on ancilla, perform controlled-SWAP, then H, then measure.
    qc.h(anc)
    for k in range(num_qubits):
        qc.cswap(anc, reg1[k], reg2[k])
    qc.h(anc)
    
    qc.measure(anc, 0)
    return qc


# 4) Builds SWAP-test circuits and compute kernel via Sampler

def build_swaptest_circuits(X_data, num_qubits=5):
    """
    For given data X_data (shape (N, F)), build SWAP-test circuits for every pair
    (i, j) with i <= j. Returns a list of circuits and a mapping list of (i, j).
    """
    circuits = []
    pair_map = []
    N = len(X_data)
    for i in range(N):
        for j in range(i, N):
            qc = swap_test_circuit(X_data[i], X_data[j], num_qubits=num_qubits)
            circuits.append(qc)
            pair_map.append((i, j))
    return circuits, pair_map

def compute_swaptest_kernel_sampler(X_data, sampler: Sampler, backend, num_qubits=5, shots=1024):
    """
    Computes an NxN kernel matrix for X_data using SWAP-test circuits executed
    with the Sampler primitive.
    """
    circuits, pair_map = build_swaptest_circuits(X_data, num_qubits=num_qubits)
    N = len(X_data)
    K = np.zeros((N, N))
    
    print(f"Transpiling {len(circuits)} circuits for backend {backend.name}...")
    tcircs = transpile(circuits, backend=backend, optimization_level=3)
    
    print("Running sampler on training circuits...")
    job_result = sampler.run(tcircs, shots=shots).result()
    # In v0.36.1, job_result is indexable. The default classical register name is 'c'.
    for idx, (i, j) in enumerate(pair_map):
        # Access counts from the default classical register 'c'
        counts = job_result[idx].data.c.get_counts()
        p0 = counts.get("0", 0) / shots
        fidelity = 2 * p0 - 1
        fidelity = max(0.0, fidelity)
        K[i, j] = fidelity
        K[j, i] = fidelity
    return K

def compute_swaptest_cross_kernel_sampler(X1, X2, sampler: Sampler, backend, num_qubits=5, shots=1024):
    """
    Computes the cross-kernel matrix between datasets X1 and X2 using SWAP-test circuits
    executed with the Sampler primitive.
    """
    circuits = []
    pair_map = []
    N1 = len(X1)
    N2 = len(X2)
    for i in range(N1):
        for j in range(N2):
            qc = swap_test_circuit(X1[i], X2[j], num_qubits=num_qubits)
            circuits.append(qc)
            pair_map.append((i, j))
    
    print(f"Transpiling {len(circuits)} cross-kernel circuits for backend {backend.name}...")
    tcircs = transpile(circuits, backend=backend, optimization_level=3)
    
    print("Running sampler on cross-kernel circuits...")
    job_result = sampler.run(tcircs, shots=shots).result()
    K_cross = np.zeros((N1, N2))
    for idx, (i, j) in enumerate(pair_map):
        counts = job_result[idx].data.c.get_counts()
        p0 = counts.get("0", 0) / shots
        fidelity = 2 * p0 - 1
        fidelity = max(0.0, fidelity)
        K_cross[i, j] = fidelity
    return K_cross


# 5) Main function: Run SWAP-test QSVM on hardware (retrieve results and run SVR)

def main_swaptest_hardware_demo(
    ibm_token="HERE_PUT_YOUR_TOKE",
    backend_name="ibm_kyiv",   # e.g., "ibm_kyiv" or "ibm_sherbrooke"
    matfile="qm7.mat",
    subset_size=8,
    num_qubits=5,
    shots=1024,
    test_size=0.25,
    random_seed=42
):
    """
    Pipeline:
      1) Loads QM7 data (23 eigenvalues per molecule)
      2) Subsamples the data
      3) Splits into training and testing sets
      4) Connects to IBM Quantum using QiskitRuntimeService and start a session
      5) Computes the training kernel via SWAP-test circuits (using Sampler)
      6) Fits an SVR using the precomputed kernel
      7) Computes the cross-kernel (test vs. train)
      8) Evaluates predictions
    """
    # Loads data
    X_all, Y_all = load_qm7_eigenvalues(matfile)
    N = len(X_all)
    print(f"Loaded {N} total molecules. Using subset_size={subset_size}.")
    
    # Subsamples data
    np.random.seed(random_seed)
    idxs = np.random.choice(N, size=subset_size, replace=False)
    X_sub = X_all[idxs]
    Y_sub = Y_all[idxs]
    
    # Splits data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, Y_sub, test_size=test_size, random_state=random_seed
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Connects to IBM Quantum using QiskitRuntimeService
    print("\nLogging into IBM Quantum using QiskitRuntimeService & starting session...")
    service = QiskitRuntimeService(token=ibm_token, channel="ibm_quantum")
    available_backends = service.backends()
    print("Backends you can use:")
    for b in available_backends:
        print("  ", b.name)
    # now picks one of those names:
    matches = [b for b in available_backends if b.name == backend_name]
    if not matches:
        raise ValueError(f"No such backend '{backend_name}'.")
    backend = matches[0]
    print(f"Using backend: {backend.name} (Qubits={backend.configuration().num_qubits})\n")
    
    # Creates a Sampler session (the classical register defaults to 'c')
    with Session(backend=backend) as session:
        sampler = Sampler(session, options={})
        
        # Computes training kernel matrix
        print("Computing training SWAP-test kernel on hardware...")
        K_train = compute_swaptest_kernel_sampler(
            X_train, sampler=sampler, backend=backend, num_qubits=num_qubits, shots=shots
        )
        
        # Fits SVR with the precomputed kernel
        svr = SVR(kernel="precomputed", C=10.0, epsilon=1.0)
        svr.fit(K_train, y_train)
        
        # Computes cross-kernel (test vs. train)
        print("Computing test vs train cross-kernel on hardware...")
        K_test = compute_swaptest_cross_kernel_sampler(
            X_test, X_train, sampler=sampler, backend=backend, num_qubits=num_qubits, shots=shots
        )
    
    # Evaluates predictions
    y_pred = svr.predict(K_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n===== SWAP TEST QSVM on real hardware (Sampler V2) =====")
    print(f"Backend: {backend.name}")
    print(f"subset_size={subset_size}, num_qubits={num_qubits}, shots={shots}")
    print(f"Train size={len(X_train)}, Test size={len(X_test)}")
    print(f"MAE = {mae:.4f}")
    print(f"R^2 = {r2:.4f}")


# 6) Command-line usage

if __name__ == "__main__":
    main_swaptest_hardware_demo(
        ibm_token="PUT_TOKEN_HERE",
        backend_name="ibm_brisbane",
        matfile="qm7.mat",
        subset_size=8,
        num_qubits=5,
        shots=1024,
        test_size=0.25,
        random_seed=42
    )
"""

# Uncomment the block below to run the SWAP-test QSVM on a full 23x23 Coulomb matrices and comment the above
# (529 features → requires 10 qubits) instead of the 23-D eigenvalues.

if __name__ == "__main__":
    # --- override loader to use the full 23x23 Coulomb matrices ---
    def load_qm7_flat(matfile="qm7.mat"):
        data = scipy.io.loadmat(matfile)
        C = data["X"]          # shape (N,23,23)
        Y = data["T"].ravel()  # shape (N,)
        X_flat = np.stack([M.flatten() for M in C], axis=0)
        return X_flat, Y

    # monkey-patch eigenvalue loader
    load_qm7_eigenvalues = load_qm7_flat

    # now calls the exact same main on 529D full-matrix with 10 qubits
    main_swaptest_hardware_demo(
        ibm_token="PUT_TOKEN_HERE",
        backend_name="ibm_brisbane",
        matfile="qm7.mat",
        subset_size=8,      
        num_qubits=10,      # ⌈log2(529)⌉ = 10
        shots=1024,
        test_size=0.25,
        random_seed=42
    )
"""
