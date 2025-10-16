# new libraries
# talked to chatgp
# https://chatgpt.com/c/68f12b38-b488-8321-89cf-fb2d66bb1e3e
# also, from the library and discussed projects https://pypi.org/project/google-genai/
# -------

from google import genai

client = genai.Client(
    vertexai=True,
    project="btibert-ba882-fall25",
    location="us-central1",
)

resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Write a haiku about data pipelines."
)
print(resp.text)