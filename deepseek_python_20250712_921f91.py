with mp.Pool(NUM_CORES) as pool:
    results = list(pool.imap(simulate_genome, args, chunksize=CHUNK_SIZE))