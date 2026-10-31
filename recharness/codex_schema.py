"""Restore optional JSON fields after strict transport validation."""
def remove_optional_nulls(value, schema):
    if schema.get("type") == "object" and isinstance(value, dict):
        required = set(schema.get("required", schema["properties"]))
        return {k: remove_optional_nulls(v, schema["properties"][k])
                for k, v in value.items() if not (v is None and k not in required)}
    if schema.get("type") == "array" and isinstance(value, list):
        return [remove_optional_nulls(v, schema["items"]) for v in value]
    return value
