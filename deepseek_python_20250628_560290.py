def calculate_revenue():
    management_fees = 0.02 * assets_under_management
    performance_fees = 0.20 * excess_returns
    licensing = 5000000  # Per investment bank
    return management_fees + performance_fees + licensing

# Conservative estimate: $2.8B/year at scale