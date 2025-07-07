import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def draw_simple_qnn_circuit(n_wires=8):
    fig, ax = plt.subplots(figsize=(14, 2 + n_wires * 0.4))

    gate_colors = {
        'H': '#81D4FA',
        'RY': '#AED581',
        'RZ': '#FFD54F',
        'CNOT': '#E57373',
        'CRZ': '#BA68C8'
    }

    wire_spacing = 1

    for wire in range(n_wires):
        ax.hlines(y=wire * wire_spacing, xmin=0, xmax=13, color='black', linewidth=1)

    def add_gate(label, wire, t):
        ax.add_patch(mpatches.FancyBboxPatch((t, wire * wire_spacing - 0.25), 0.8, 0.5,
                                              boxstyle="round,pad=0.02", linewidth=1.2,
                                              edgecolor='black', facecolor=gate_colors[label]))
        ax.text(t + 0.4, wire * wire_spacing, label, ha='center', va='center', fontsize=8)

    def add_cnot(control, target, t):
        ax.plot(t + 0.4, control * wire_spacing, 'ko', markersize=5)
        ax.plot(t + 0.4, target * wire_spacing, 'wo', markersize=10, markeredgecolor='black')
        ax.vlines(x=t + 0.4, ymin=min(control, target) * wire_spacing, ymax=max(control, target) * wire_spacing, colors='black')

    def add_crz(control, target, t):
        ax.plot(t + 0.4, control * wire_spacing, 'ko', markersize=5)
        ax.add_patch(mpatches.FancyBboxPatch((t, target * wire_spacing - 0.25), 0.8, 0.5,
                                              boxstyle="round,pad=0.02", linewidth=1.2,
                                              edgecolor='black', facecolor=gate_colors['CRZ']))
        ax.text(t + 0.4, target * wire_spacing, 'CRZ', ha='center', va='center', fontsize=6)
        ax.vlines(x=t + 0.4, ymin=min(control, target) * wire_spacing, ymax=max(control, target) * wire_spacing, colors='black')

    for wire in range(n_wires):
        add_gate('H', wire, 1)
        add_gate('RY', wire, 2)
        add_gate('RZ', wire, 4)

    for i in range(n_wires):
        add_cnot(i, (i + 1) % n_wires, 3)
        add_crz(i, (i + 2) % n_wires, 5)
        add_cnot(i, (i + 1) % n_wires, 6)

    for wire in range(n_wires):
        add_gate('RY', wire, 8)
        add_gate('RZ', wire, 10)

    for i in range(n_wires):
        add_cnot(i, (i + 1) % n_wires, 9)
        add_cnot(i, (i + 1) % n_wires, 11)

    ax.set_ylim(-1, n_wires * wire_spacing)
    ax.set_xlim(0, 13)
    ax.axis('off')
    plt.title(f'Simplified Quantum Circuit (n_wires={n_wires})', fontsize=12)
    plt.tight_layout()
    plt.show()

draw_simple_qnn_circuit(n_wires=8)
