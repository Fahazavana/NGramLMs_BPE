class Tokenizer:
    def __init__(self, file_name):
        self.file_name = file_name

    def __iter__(self):
        """
            Read the corpus and transform it into a
            List of characters.
            <s>: starting of a sentence
            </s>:end of a sentence
        Args:
            file_name: path to file containing the corpus

        Returns:
            token: List of all characters
        """
        try:
            with open(self.file_name) as file:
                for line in file:
                    yield from ['<s>', ' ']
                    yield from line.strip()  # remove new line and change it into a ending maker
                    yield from ['</s>', '\n']

        except FileNotFoundError:
            print(f"File not found at {self.file_name}")

    def build_ngram(self, order):
        ngrams = set()
        tokens = list(self)
        print(tokens)
        L = len(tokens) - order
        for i in range(L):
            current = "".join(tokens[i * order:(i + 1) * order])
            ngrams = ngrams.union(current)
        return ngrams
