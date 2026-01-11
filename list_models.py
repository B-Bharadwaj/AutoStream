from google import genai

client = genai.Client()

print("generateContent models:")
for m in client.models.list():
    if "generateContent" in getattr(m, "supported_actions", []):
        print(m.name)

print("\nembedContent models:")
for m in client.models.list():
    if "embedContent" in getattr(m, "supported_actions", []):
        print(m.name)
