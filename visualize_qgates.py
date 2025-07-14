from qiskit import QuantumCircuit
from qiskit.circuit.library import U3Gate, CU3Gate
import matplotlib.pyplot as plt

n_qubits = 4
depth = 3

qc = QuantumCircuit(n_qubits)
theta, phi, lam = 0.3, 0.2, 0.1

for _ in range(depth):
    for q in range(n_qubits):
        qc.append(U3Gate(theta, phi, lam), [q])
    for q in range(n_qubits):
        control = q
        target = (q + 1) % n_qubits
        qc.append(CU3Gate(theta, phi, lam), [control, target])

# Plot and show explicitly
fig = qc.draw('mpl', fold = -1)
plt.show()
