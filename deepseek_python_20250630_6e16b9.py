np.savez('arrow_data.npz', 
         x=sim.x, S=sim.S, V=sim.𝒱, dSdt=dSdt, 
         params={'τ0': sim.τ0, 'σS': sim.σS, 'D': sim.D, 'λ': sim.λ})