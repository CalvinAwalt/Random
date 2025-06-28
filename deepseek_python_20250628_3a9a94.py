from web3 import Web3
w3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/YOUR_PROJECT_ID'))

formula_hash = Web3.sha3(text=json.dumps(FORMULA_REGISTRY))
tx = w3.eth.send_transaction({
    'to': '0x0000000000000000000000000000000000000000',
    'value': 0,
    'data': formula_hash
})
print(f"Formulas anchored in Ethereum tx: {tx.hex()}")