from qiskit import QuantumCircuit  
qc = QuantumCircuit(4)  
qc.append(CalvinOperator(), [0,1,2,3])  # Hardware test  