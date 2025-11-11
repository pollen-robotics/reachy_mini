"""Test the schema conversion."""
import sys
sys.path.insert(0, '/Users/lauras/Desktop/laura/reachy_mini/reachy_mini_conversation_app/src')

from google.generativeai import protos

# Simple test schema
test_schema = {
    "type": "object",
    "properties": {
        "direction": {
            "type": "string",
            "enum": ["left", "right", "up", "down"],
        }
    },
    "required": ["direction"]
}

def convert_json_schema_to_gemini(schema: dict) -> dict:
    """Convert JSON Schema format to Gemini's schema format using Type enums."""
    if not isinstance(schema, dict):
        return schema

    # Mapping from JSON Schema type strings to Gemini Type enums
    type_mapping = {
        "string": protos.Type.STRING,
        "number": protos.Type.NUMBER,
        "integer": protos.Type.INTEGER,
        "boolean": protos.Type.BOOLEAN,
        "object": protos.Type.OBJECT,
        "array": protos.Type.ARRAY,
    }

    result = {}
    for key, value in schema.items():
        # Convert 'type' field from string to Gemini Type enum
        if key == "type" and isinstance(value, str):
            result[key] = type_mapping.get(value, protos.Type.STRING)
        # Recursively process nested dicts
        elif isinstance(value, dict):
            result[key] = convert_json_schema_to_gemini(value)
        # Recursively process lists
        elif isinstance(value, list):
            result[key] = [convert_json_schema_to_gemini(item) if isinstance(item, dict) else item for item in value]
        else:
            result[key] = value
    return result

print("Original schema:")
print(test_schema)
print("\nConverted schema:")
converted = convert_json_schema_to_gemini(test_schema)
print(converted)

# Try to use it with Gemini
import google.generativeai as genai

test_function = {
    "name": "move_head",
    "description": "Move head in a direction",
    "parameters": converted
}

print("\nTesting with Gemini...")
try:
    model = genai.GenerativeModel(
        "models/gemini-1.5-pro-latest",
        tools=[test_function]
    )
    print("✓ Schema accepted by Gemini!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
