"""Code for calling the training of the model."""

from sys import argv
import json
import pickle
from pathlib import Path
from collections import defaultdict, Counter
import logging 
from TextGeneration.utils.files import json_to_schema, read_dir
from TextGeneration.utils.preprocessor import Preprocessor
from TextGeneration.utils.schemas import TrainingInputSchema

logging.basicConfig(level=logging.INFO)

class NGramModel:
    """Class for training an n-gram language model."""

    def __init__(self, max_n_grams_to_generate: int):
        """
        Initialize the NGramModel. Also initializes the counts and probabilities of the ngrams 
        and the minimum count for an ngram to be considered (which is 6).

        :param max_n_grams_to_generate: The maximum number of n-grams to generate (taken from the user, given)

        """
        self.max_n_grams_to_generate = max_n_grams_to_generate
        self.ngram_counts = defaultdict(lambda: defaultdict(Counter))
        self.ngram_probs = defaultdict(lambda: defaultdict(dict))
        self.min_count = 6 #Since we don't want to generate ngrams with less than 6 occurences


    def process_text(self, line: str) -> None:
        """Process a line of text to extract n-grams."""
        line = Preprocessor.clean(text=line)

        #If the line is empty, continue
        if not line:
            logging.warning("Empty line encountered in the training data.")
            return
        
        words = line.split(" ")
        # If the line is less than 1 word, continue
        if len(words) < 1:
            logging.warning("Line with less than 1 word encountered in the training data.")
            return 

        logging.info(f"Processing line: {line}")
        # Generate all ngrams upto max_n_grams_to_generate
        for n_gram_length in range(1, self.max_n_grams_to_generate + 1):
            # check if the ngram length is greater than the number of words in the line
            if n_gram_length > len(words):
                logging.warning(f"Ngram length {n_gram_length} is greater than the number of words in the line.")
                continue

            # Generate the ngrams dict
            for position in range(len(words) - n_gram_length + 1):
                context = tuple(words[position:position+n_gram_length-1]) if n_gram_length > 1 else ()
                next_words = words[position+n_gram_length-1]
                self.ngram_counts[n_gram_length][context][next_words] += 1
                logging.debug(f"Ngram: {n_gram_length}, Context: {context}, Next words: {next_words}")

    def calculate_probabilities(self) -> None:
        for n_gram_length in range(1, self.max_n_grams_to_generate + 1):
            logging.info(f"Calculating probabilities for n_gram_length: {n_gram_length}")
            for context, next_words in self.ngram_counts[n_gram_length].items():
                # Filter out n-grams with fewer than min_count occurrences
                filtered_next_words = { word: count 
                                        for word, count in next_words.items() 
                                        if count >= self.min_count
                }
                
                if filtered_next_words:
                    logging.debug(f"Filtered next words: {filtered_next_words}")
                    total_count = sum(filtered_next_words.values())
                    logging.debug(f"Total count: {total_count}")
                    # Calculate probabilities
                    self.ngram_probs[n_gram_length][context] = {word: count / total_count 
                                                   for word, count in filtered_next_words.items()}
        logging.debug(f"Probabilities: {dict(self.ngram_probs)}")

    def save(self, file_path: str) -> None:
        """Save the model to a file."""
        # Convert defaultdicts to regular dicts for serialization
        ngram_probs_dict = {}
        for n, contexts in self.ngram_probs.items():
            ngram_probs_dict[n] = {context: dict(next_words) for context, next_words in contexts.items()}
        
        model_data = {
            'max_n_gram': self.max_n_grams_to_generate,
            'ngram_probs': ngram_probs_dict
        }
        logging.debug(f"Model data: {model_data}")
        # Use pickle for saving the model data
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)


def main_train(file_str_path: str) -> None:
    """
    Call for training an n-gram language model.

    Do not modify its signature.
    You can modify the content.

    :param file_str_path: The path to the JSON that configures the training
    :return: None
    """
    # Reading input data
    training_schema = json_to_schema(
        file_str_path=file_str_path, 
        input_schema=TrainingInputSchema
    )
    logging.debug(f"Training schema: {training_schema}")
    
    model = NGramModel(max_n_grams_to_generate=training_schema.max_n_gram)
        
    # Process each line in the training data
    logging.info("Processing training data...")
    for training_line in read_dir(dir_path=training_schema.input_folder):
        logging.debug(f"Processing training line: {training_line}")
        model.process_text(training_line)
    
    logging.debug("Calculating probabilities...")
    model.calculate_probabilities()

    # Save the model
    logging.info("Saving model...")
    model.save(training_schema.trained_model)
    logging.debug("Model saved successfully.")

if __name__ == "__main__":
    main_train(file_str_path=argv[1])
