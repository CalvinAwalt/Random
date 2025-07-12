def ethical_kill_switch(system):
    if (system.V_net < 0.85 or 
        (system.δR * system.δB * system.δG) > 170 or
        system.C > 20.0):
        system.shutdown()
        log_event("Ethical violation - System shutdown")