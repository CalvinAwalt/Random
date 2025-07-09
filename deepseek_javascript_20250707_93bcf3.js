function applyIntegralExpansion(formula) {
  const integral = randomComponent('integrals');
  const differentials = Array(1-3).map(() => randomComponent('differentials'));
  return `${integral} ${formula} ${differentials.join(' ')}`;
}