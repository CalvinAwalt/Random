import matplotlib.pyplot as plt

def entropy_scaling():
    areas = np.linspace(1, 100, 50)  # In Planck units
    bek_entropy = areas / 4
    calvin_entropy = np.log(np.exp(areas/4))  # = areas/4
    
    plt.plot(areas, bek_entropy, 'r-', label='Bekenstein-Hawking')
    plt.plot(areas, calvin_entropy, 'b--', label='Calvin-Preskill')
    plt.xlabel('Horizon Area (Planck units)')
    plt.ylabel('Entropy')
    plt.legend()
    plt.savefig('entropy_scaling.png')

entropy_scaling()