from langchain_core.prompts import load_prompt

loaded_prompt = load_prompt("./02_model_io/prompt.json")
print(loaded_prompt.format(product="iPhone"))