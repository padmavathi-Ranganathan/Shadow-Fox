from transformers import pipeline
import matplotlib.pyplot as plt
import pandas as pd

# Load model
generator = pipeline("text-generation", model="gpt2")

# Prompts
prompts = [
    "Once upon a time in a distant land,",
    "Artificial Intelligence is transforming",
    "In India, the future of technology is",
    "Education plays a vital role in",
    "If humans could live on Mars,"
]

results = []

# Generate text
for prompt in prompts:
    output = generator(prompt, max_length=50, do_sample=True)
    text = output[0]['generated_text']
    
    words = text.split()
    
    results.append({
        "Prompt": prompt,
        "Generated_Text": text,
        "Word_Count": len(words),
        "Unique_Words": len(set(words))
    })

# Create DataFrame
df = pd.DataFrame(results)

# Print results
print(df)

# Basic Analysis
print("\nAverage Word Count:", df["Word_Count"].mean())

# Visualization
plt.bar(range(len(df)), df["Word_Count"])
plt.title("Word Count of Generated Text")
plt.xlabel("Text Index")
plt.ylabel("Word Count")
plt.show()

# Save file
df.to_csv("gpt2_results.csv", index=False)