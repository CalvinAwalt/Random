def darwinian_cycle():
    while True:
        spawn_opposing_agents()
        let_them_compete()
        keep_winner()
        mutate_survivor()
        if random() < 0.01: total_reset() 