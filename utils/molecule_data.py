import pennylane as qml
import numpy as np
from itertools import combinations

def generate_lie_algebra_operator_pool(singles, doubles, triples, op_times, num_qubits):
    operator_pool = []
    for single in singles:
        for time in op_times:
            operator_pool.append(qml.SingleExcitation(time, wires=single))
    for double in doubles:
        for time in op_times:
            operator_pool.append(qml.DoubleExcitation(time, wires=double))
    for triple in triples:
        for time in op_times:
            operator_pool.append(qml.TripleExcitation(time, wires=triple))
    for w in range(num_qubits):
        for time in op_times:
            operator_pool.append(qml.Identity(wires=w))
    return operator_pool

def generate_ucc_operator_pool(singles, doubles, op_times, num_qubits):
    """Generate the UCC operator pool including single and double excitations."""
    operator_pool = []
    for single in singles:
        for time in op_times:
            operator_pool.append(qml.SingleExcitation(time, wires=single))
            operator_pool.append(qml.SingleExcitationMinus(time, wires=single))  # UCC single excitation
    for double in doubles:
        for time in op_times:
            operator_pool.append(qml.DoubleExcitation(time, wires=double))
            operator_pool.append(qml.DoubleExcitationMinus(time, wires=double))  # UCC double excitation
    return operator_pool

def generate_molecule_data(molecules="H2", use_ucc=True, use_lie_algebra=True):
    datasets = qml.data.load("qchem", molname=molecules)
    op_times = np.sort(np.logspace(-2, 0, num=8)) / 160
    molecule_data = {}
    for dataset in datasets:
        molecule = dataset.molecule
        num_electrons, num_qubits = molecule.n_electrons, 2 * molecule.n_orbitals
        singles, doubles = qml.qchem.excitations(num_electrons, num_qubits)
        triples = [tuple(sorted(set(sum(combo, ())))) for combo in combinations(singles, 3)]
        triples = list(set(triple for triple in triples if len(triple) == 6))

        if use_ucc:
            operator_pool = generate_ucc_operator_pool(singles, doubles, op_times, num_qubits)
        elif use_lie_algebra:
            operator_pool = generate_lie_algebra_operator_pool(singles, doubles, triples, op_times, num_qubits)
        else:
            operator_pool = [qml.Identity(wires=w) for w in range(num_qubits)]

        molecule_data[dataset.molname] = {
            "op_pool": operator_pool,
            "num_qubits": num_qubits,
            "hf_state": dataset.hf_state,
            "hamiltonian": dataset.hamiltonian,
            "expected_ground_state_E": dataset.fci_energy,
        }
    return molecule_data
