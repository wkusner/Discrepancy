import json, os
def load_critical():
    """Dictionary of verified critical configurations (see critical_configurations.json)."""
    with open(os.path.join(os.path.dirname(__file__), "critical_configurations.json")) as f:
        return {c["name"]: c for c in json.load(f)["classes"]}
