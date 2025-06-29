def calvin_encrypt(message, L=3):
    k = np.log(3)/np.log(2)
    key = int(np.exp(k * L)) % PRIME
    return aes_encrypt(message, key)