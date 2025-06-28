from qiskit_ibm_runtime import QiskitRuntimeService
service = QiskitRuntimeService(channel="ibm_quantum")
backend = service.backend("ibm_kyiv")  # Future 1000+ qubit system