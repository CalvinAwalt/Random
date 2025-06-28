from qiskit import IBMQ
provider = IBMQ.load_account()
backend = provider.get_backend('ibm_quantum')