"""Code for calling the generating a text."""

from sys import argv
import pickle
import random
import logging
from TextGeneration.utils.files import json_to_schema, schema_to_json
from TextGeneration.utils.preprocessor import Preprocessor
from TextGeneration.utils.schemas import InputSchema, OutputSchema


def load_model(file_path: str) -> dict:
    """
    Load the trained model from a file.
    
    :param file_path: The path to the file containing the trained model
    :return: The trained model
    """
    try:
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise


def predict_next_word(
        model_data: dict, 
        context: list, 
        max_n_gram: int, 
        use_top_candidate: int = 1
    ) -> str | None:
    """
    Predict the next word based on the given context.

    :param model_data: The trained model data
    :param context: List of previous words
    :param max_n_gram: Maximum n-gram size to use for prediction
    :param use_top_candidate: How many top candidates to consider
        
    :return: The predicted next word or None if no prediction available
    """
    ngram_probs = model_data['ngram_probs']
    model_max_n_gram = model_data['max_n_gram']
    
    # Use the minimum of the model's max n-gram and the requested max n-gram
    max_n_gram = min(model_max_n_gram, max_n_gram)
    
    # then start with the largest possible n-gram size based on context
    n = min(max_n_gram, len(context) + 1)
    
    # Simplified backoff to use largest possible n-gram size
    while n > 0:
        # If the current n-gram size doesn't exist
        if n not in ngram_probs:
            n -= 1
            continue
            
        # For n-gram, use the last n-1 words as context
        if n == 1:
            context_n = ()
        else:
            context_n = tuple(context[-(n-1):])
        
        if context_n not in ngram_probs[n]:
            n -= 1
            continue
            
        next_words = ngram_probs[n][context_n]
        
        if not next_words:
            n -= 1
            continue
            
        # Sort by probability
        sorted_words = sorted(next_words.items(), key=lambda x: x[1], reverse=True)
        
        # Get the top candidates (limit to available candidates)
        top_candidates = sorted_words[:min(use_top_candidate, len(sorted_words))]
        
        # If we have only one candidate or only considering top 1
        if use_top_candidate == 1 or len(top_candidates) == 1:
            return top_candidates[0][0]
        else:
            # Randomly select among top candidates based on their probabilities
            words, probs = zip(*top_candidates)
            # Normalize probabilities
            total_prob = sum(probs)
            normalized_probs = [p/total_prob for p in probs]
            return random.choices(words, weights=normalized_probs, k=1)[0]
    
    return None


def generate_text(
        model_data: dict, 
        input_text: str, 
        max_n_gram: int, 
        use_top_candidate: int = 1, 
        max_words: int = 50
    ) -> str:
    """
    Generate text based on the input text and trained model.
    
    :param model_data: The trained model data
    :param input_text: The starting text
    :param max_n_gram: Maximum n-gram size to use for prediction
    :param use_top_candidate: How many top candidates to consider
    :param max_words: Maximum number of words to generate
        
    :return: The generated text
    """
    # Clean input text
    text = Preprocessor.clean(input_text)
    
    # If text is empty, start with an empty context
    if text:
        words = text.split()
    else:
        words = []
    
    # Generate words
    words_generated = 0
    while words_generated < max_words:
        # Get the context
        context = words  # Use all available words
        next_word = predict_next_word(model_data, context, max_n_gram, use_top_candidate)
        
        # If no prediction available or end of string predicted
        if next_word is None:
            break
        
        # Add next word
        words.append(next_word)
        words_generated += 1
    
    if not words:
        return ""
        
    # Join words and ensure no leading/trailing spaces
    generated_text = ' '.join(words).strip()
    return generated_text



def main_generate(file_str_path: str) -> None:
    """
    Call for generating a text.

    Do not modify its signature.
    You can modify the content.

    :param file_str_path: The path to the JSON that configures the generation
    :return: None
    """
    # Reading input data
    input_schema = json_to_schema(file_str_path=file_str_path, input_schema=InputSchema)

    # Load the model
    model_data = load_model(input_schema.trained_model)
    
    generated_texts = []
    for input_text in input_schema.texts:
        generated_text = generate_text(
                model_data,
                input_text,
                max_n_gram=input_schema.max_n_gram,
                use_top_candidate=input_schema.use_top_candidate
            )
        generated_texts.append(generated_text)

    # Printing generated texts
    output_schema = OutputSchema(generated_texts=generated_texts)
    schema_to_json(file_path=input_schema.output_file, schema=output_schema)


if __name__ == "__main__":
    main_generate(file_str_path=argv[1])
