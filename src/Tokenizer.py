from typing import List, Generator, Tuple
class Tokenizer:
    """
    Tokenizer class:
        - Token character level as N-gramm
    """
    def __init__(self, start="<", end=">"):
        self.start = start
        self.end = end
        
    def charSentenceTokenizer(self, sentence, order)->Generator:
        """
        Tokenize a sentence into charcters and include the start and end sentence marker.
        Args:
            sentence: sentence to be tokenized
            order: N-gram order
        Returns:
            generator: generator of Ngram of tokens
        """
        sentence = [self.start] + list(sentence.strip()) + [self.end]
        return self.__gen__(sentence, order)

    def __gen__(self, tokens:List, order:int):
        """
        N-Gram generator for a given token
        Args:
            tokens: List of the tokens (words or characters)
            order: N-Gram order
        Returns:

        """
        if order <= 0:
            raise ValueError("Order must be greater than 0")
        nbr_ngram = len(tokens) - order + 1
        for i in range(nbr_ngram):
            ngram = tuple(tokens[i : i + order])
            if ngram:
                yield ngram

    def wordTokenizer(self, sentence:str)->List[str]:
        """
        Tokenize a sentence into words and include the start and end sentence marker.
        Args:
            sentence: the sentence to be tokenized
        Returns:
            (list): list of tokens
        """
        return [self.start] + sentence.strip().split() + [self.end]