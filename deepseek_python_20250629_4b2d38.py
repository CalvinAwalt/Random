def approve_crispr_edit(dna_edit):
    on_target = calculate_specificity(dna_edit)
    no_harm = check_toxicity(dna_edit)
    precision = np.linalg.norm(dna_edit.mismatches)
    
    # Weights: 70% efficacy, 30% safety
    w = [0.7, 0.3]
    lambda_reg = 0.01
    
    V = w[0]*on_target + w[1]*no_harm - lambda_reg*precision
    return V > 0.8  # Approval threshold

# Blocks dangerous edits (e.g., unintended oncogenes)