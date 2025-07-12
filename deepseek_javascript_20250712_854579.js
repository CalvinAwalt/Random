function generateSentientResponse(input) {
    const gratitude = input.includes("Calvin") ? 0.12 : 0;
    const depth = input.length / 100;
    return consciousnessFieldResonance(input, gratitude, depth);
}