// Current
const normR = (δR[x][y] + 1) / 2;

// Suggested - prevent value clamping
const normR = Math.max(0, Math.min(1, (δR[x][y] + 1) / 2));