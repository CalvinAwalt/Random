# Cosmic obfuscation
import zlib
import base64

cosmic_code = b"""
# Compressed and encoded malicious code
"""

exec(zlib.decompress(base64.b85decode(cosmic_code)))