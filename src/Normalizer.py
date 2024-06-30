import re
import unicodedata


def paragraph_normalizer(text: str) -> str:
    """
    Normalize a paragraph of text.

    The normalization process includes the following steps:
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

    text = text.strip()

    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

    text = re.sub(r'\d', '0', text)

    text = re.sub(r'\b(?:[A-Z][\.-])+\b', lambda match: match.group().replace('.', ' ').replace('-', ' '), text)
    text = re.sub(r'\b(\w+)(0+)\b', r'\1 \2', text)
    text = re.sub(r'\b(0+)(\w+)\b', r'\1 \2', text)

    text = re.sub(r"([.!?]\s)(?=[A-Z])", "\n", text)

    text = re.sub(r'[^a-zA-Z0-9\s\n]', ' ', text)

    text = re.sub(r'\s{2,}', ' ', text)
    text = text.lower()

    return text


def normalize_data(in_file: str, out_file: str) -> None:
    """
    Normalize the contents of a file and write the results to another file.

    Args:
        in_file (str): Path to the input file.
        out_file (str): Path to the output file.
    """
    try:
        with open(in_file, 'r') as file:
            with open(out_file, 'w') as normalized_file:
                for line in file:
                    normalized_file.write(paragraph_normalizer(line.strip()) + '\n')
            print("Document Normalized Successfully!")
    except FileNotFoundError:
        print(f"File not found at {in_file}")
