from qiskit import QuantumCircuit
from qiskit.circuit.library import U3Gate
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt

def build_qnn_circuit(n_qubits=4, q_depth=3, theta=0.3, phi=0.2, lam=0.1):
    qc = QuantumCircuit(n_qubits)

    for depth in range(q_depth):
        # U3 gates
        for q in range(n_qubits):
            qc.u(theta, phi, lam, q)

        # CU3 gates — circular
        for q in range(n_qubits):
            control = q
            target = (q + 1) % n_qubits
            qc.append(U3Gate(theta, phi, lam).control(1), [control, target])

    return qc


if __name__ == "__main__":
    qc = build_qnn_circuit(n_qubits=4, q_depth=3)

    fig = circuit_drawer(qc, output='mpl', style='iqx', fold = -1)
    plt.tight_layout()
    plt.savefig("qnn_circuit.png", dpi=300)
    plt.show()
