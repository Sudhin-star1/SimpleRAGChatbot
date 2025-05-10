from llm_answer import ask_llm

# Example context and question
context = "The sun is the star at the center of the solar system. It provides light and heat to Earth."
question = "What is the sun and why is it important?"

# Get response from the local Mistral model
response = ask_llm(question, context)

# Print the result
print("LLM Answer:", response)
