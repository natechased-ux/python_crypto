import json

# Path to your corrupted file
input_file = "whale_history_charlie.json"

# Try to recover what we can
with open(input_file, "r") as f:
    content = f.read()

# Try trimming the file to the last valid JSON
try:
    while content:
        try:
            data = json.loads(content)
            print("✅ Successfully recovered JSON!")
            with open(input_file, "w") as f:
                json.dump(data, f)
            break
        except json.JSONDecodeError:
            content = content[:-1]
except Exception as e:
    print("❌ Recovery failed:", e)
