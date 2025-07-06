def detect_threat(ip):
    if ip in adversarial_ranges:
        deploy_honeypot(ip)
        alert_authorities(ip)
        return ThreatLevel.RED
    return ThreatLevel.GREEN