import re
import string
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

    # Normalize diacritics to simple characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

    acrs_and_abvs = re.findall(r"\b([A-Z]{2,}|[A-Z]+[0-9]+|[A-Z]+[a-z]+[A-Z]|[a-z]+[0-9]+)", text)
    for acr_and_abv in acrs_and_abvs:
        if any(char in string.punctuation for char in acr_and_abv) and len(acr_and_abv) > 1:
            text = text.replace(acr_and_abv, " ".join(acr_and_abv))

    text = re.sub(r"([.!?]\s)(?=[A-Z])", "\n", text)

    text = re.sub(r"\d", "0", text)
    text = re.sub(r"[^\w\s\n]", " ", text)

    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"_", " ", text)

    # Convert all text to lowercase
    text = text.lower()

    return text


def normalize_data(in_file: str, out_file: str)->None:
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


# Example usage
if __name__ == "__main__":
    input_file = "input.txt"
    output_file = "output.txt"
    normalize_file(input_file, output_file)
