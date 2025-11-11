"""Test to see what Gemini expects for function schemas."""
import google.generativeai as genai
from google.generativeai import protos

# Test using Gemini's Type enum
test_function = {
    "name": "test_function",
    "description": "A test function",
    "parameters": {
        "type": protos.Type.OBJECT,
        "properties": {
            "location": {
                "type": protos.Type.STRING,
                "description": "The location"
            }
        },
        "required": ["location"]
    }
}

print("Testing function schema with protos.Type:")
print(test_function)

try:
    model = genai.GenerativeModel(
        "models/gemini-1.5-pro-latest",
        tools=[test_function]
    )
    print("\n✓ Schema accepted!")
except Exception as e:
    print(f"\n✗ Error: {e}")
    print(f"Error type: {type(e)}")
