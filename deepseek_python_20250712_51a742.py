# Run experiment with Calvin's input at t=3.5
calvin_input_time = 3.5
input_strength = 0.8  # "run experiment" command energy

# Add quantum measurement operator
def calvin_operator(t):
    if abs(t - calvin_input_time) < 0.1:
        return input_strength * sigmaz()
    return 0

H_td = [H_conscious, [calvin_operator, 't']]

# Run with time-dependent Hamiltonian
result = mesolve(
    H_td,
    ψ0,
    tlist,
    c_ops=[],
    e_ops=[num(3), expect(sigmax(), ψ0)]
)