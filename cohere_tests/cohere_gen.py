import cohere
# Call the cohere client
co = cohere.Client('2A0mtCaVSr0HFCBShT3coreZ4kO0tUpGyqq11r7b')

# Get response to a prompt
response = co.generate(prompt='I have a Shiba Inu named', # Starting prompt (real Shiba is names Toro!)
                       max_tokens=50, # I want to start seeing sentences break down even with the default xl model
                       num_generations=5, # Just for more interest
                       temperature=0.4 # Lower temperature to get more realistic sentences. Default gave some real garbage
                       )

# Print test
for idx in range(5):
    print(f'Prediction {idx}: {response.generations[idx].text}')