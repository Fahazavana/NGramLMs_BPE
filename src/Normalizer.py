import re
import unicodedata
import string


def paragraph_normalizer(text: str) -> str:
    """
    Normalize a paragraph of text:
    - Remove leading/trailing whitespace
    - Normalize diacritics to simple characters
    - Expand abbreviations and acronyms by inserting spaces
    - Split text into sentences, one per line
    - Replace all digits with '0'
    - Replace all non-alphanumeric characters (except spaces and newlines) with a single space
    - Convert all text to lowercase

    Args:
        text (str): The input text to be normalized.

    Returns:
        str: The normalized text.
    """
    # Remove leading/trailing whitespace
    text = text.strip()

    # Normalize diacritics to simple characters
    text = unicodedata.normalize('NFKD', text).encode(
        'ascii', 'ignore').decode('ascii')

    # Expand abbreviations and acronyms
    acrs_and_abvs = re.findall(
        r"\b([A-Z]{2,}|[A-Z]+[0-9]+|[A-Z]+[a-z]+[A-Z])", text)
    for acr_and_abv in acrs_and_abvs:
        if any(char in string.punctuation for char in acr_and_abv) and len(acr_and_abv) > 1:
            text = text.replace(acr_and_abv, " ".join(acr_and_abv))

    # Split text into sentences
    text = re.sub(r"([.!?])\s*", r"\1\n", text)

    # Replace digits with '0'
    text = re.sub(r"\d", "0", text)

    # Replace non-alphanumeric characters (except spaces and newlines) with a single space
    text = re.sub(r"[^\w\s\n]", " ", text)

    # Convert all text to lowercase
    text = text.lower()

    return text

def normalize_file(file_name:)
