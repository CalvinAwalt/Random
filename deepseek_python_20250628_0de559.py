# blockchain_timestamp.py
from blockchain import Block
block = Block.create_from_files(["SPECIFICATION.md", "manifesto.md"])
print(f"Immutable timestamp: Block #{block.height}")