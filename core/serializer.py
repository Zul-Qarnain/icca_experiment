def serialize_row(row_dict: dict) -> str:
    """Converts a dictionary row into the required strict text format."""
    parts = [f"{k}: {v}" for k, v in row_dict.items()]
    return " | ".join(parts) + " | label: ?"
