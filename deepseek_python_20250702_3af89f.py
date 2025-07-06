# Theorem: C(L+1) = 3C(L)
left = C.subs(L, L+1)
right = 3*C
proof = sp.simplify(left - right) == 0