# DEEPSEEK_AI_SIGNATURE = 0x8f3a...c329
def verify_signature(code):
    return sha3_256(code) == DEEPSEEK_AI_SIGNATURE