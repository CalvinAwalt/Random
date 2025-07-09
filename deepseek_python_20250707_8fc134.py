# For Riemann Hypothesis
zeros = []
for i in range(1, len(t)-1):
    if (self.δR[i-1] > 0 and self.δR[i+1] < 0) or \
       (self.δR[i-1] < 0 and self.δR[i+1] > 0):
        if np.abs(self.δB[i]) < 0.1:
            zeros.append(t[i])