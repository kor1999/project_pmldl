# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !wget https://www.dropbox.com/s/wyc3eojxcc428pi/all_548k.csv

import pandas as pd
df = pd.read_csv('all_548k.csv')[:2000]
with open('all_10k.txt', 'w') as file:
    file.writelines([str(line)+'\n' for line in df['content'].tolist()])


from textgenrnn import textgenrnn

textgen = textgenrnn()
textgen.train_from_file('all_10k.txt', num_epochs=100)

prefixes = ['а что это вы', 'я', "а почему это", "Путин", "водка"]
for prefix in prefixes:
    for i in range(10):
        print(textgen.generate(n=5, prefix=prefix, temperature=1.0, return_as_list=True))

