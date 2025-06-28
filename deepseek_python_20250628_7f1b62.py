def process_query(input):
    creative = generate_options(input)        # Red vertex
    critical = evaluate_risks(creative)       # Blue vertex
    executive = apply_constraints(critical)   # Gold vertex
    return executive