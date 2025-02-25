# Simple Text Generator

## Goals
  - [x] Create a code for training an $n$-gram language model:
    - [x] Level 1: The size of the $n$-gram must be 2.
    - [x] Level 2: The size of the $n$-gram must be 3, with a backup for $n$-grams of size $2$. In other
    words, if the largest $n$-gram does not exist, it should try with a smaller one 
    (i.e., a simplified backoff approach.)
  - [x] Create a code for reading the language model and generate text.
    - [x] Level 1: The code must generate the text selecting the most probable word, i.e. top $x=1$ words.
    - [x] Level 2: The user should be able to define from how many top $x$ words, the generator.
    can select a word to generate a text. This will be defined through the field `use_top_candidate` in the `InputSchema` 
  - [x] The text generation can be triggered without defining a starting text (i.e., empty string). 

### Optional goals
- [x] Let the user define the size of the $n$-gram language model to train.
- [x] Let the user define the maximum size of the $n$-gram to use during the generation.
- [x] Implement for any $n$-gram language model a simple backoff approach 

## Notes:

- Use a dictionary to store the $n$-grams and their counts/probabilities. Enabled easier backoff approach.
- For the backoff approach, I used a while loop to try to find the largest $n$-gram that exists in the model. If no $n$-gram is found, it will try with a smaller one.
- For `use_top_candidate`>1, used `random.choices` to select the next word with weights given from normalizing the probabilities of the top candidates.
- For the generation, I used a while loop to generate the text. It will keep generating words until the number of words generated is greater than or equal to the maximum number of words to generate.
- Changed the read_dir function in utils/files.py to read any kind of files in a directory. From `*.txt` to `*`.