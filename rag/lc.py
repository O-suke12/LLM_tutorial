import langchain

llm = langchain.OpenAI(temperature=0.5, max_tokens=100)
output = llm.predict("日本の総理大臣は誰ですか？")

print(output)
