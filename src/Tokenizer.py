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
                    yield from line.strip() 
                    yield from ['</s>']

        except FileNotFoundError:
            print(f"File not found at {self.file_name}")

    def __build_ngram(self, tokens, order):
        ngrams = set()
        L = len(tokens) - order
        for i in range(L):
            current = tuple(tokens[i * order:(i + 1) * order])
            ngrams = ngrams.union({current})
        return ngrams

    def build_ngram(self, orders):
        ngrams = {}
        tokens = list(self.__iter__())
        for order in range(1, orders+1):
            ngrams[order] = self.__build_ngram(tokens, order)
        return ngrams

class SentenceTokenizer:
    def __init__(self, text):
        self.text = self.__tokenize(text)
    
    def __tokenize(self, text):
        seq = ['<s>', ' ']
        seq.extend(text.strip())
        return seq + ['</s>']
    
    def get_ngram(self, order):
        n = len(self.text)
        ngram = []
        for i in range(n):
            _tmp = tuple(self.text[i*order:(i+1)*order])
            if _tmp == ():
                return ngram
            ngram.append(_tmp)
        return ngram
