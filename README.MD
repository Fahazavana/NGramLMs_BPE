# Tri-Gram Language Model and Byte-Pair Encodding: Language Identification and Similarity

## Description
This project focuses on developing a language identification and similarity analysis system.
To identify the language used in a given sentence, we build a character-level trigram language model and use perplexity to determine the most likely language.
Furthermore, this can also be used to see how similar two languages are.
This latter is also analysed by performing a Byte-Pair Encoding on each language and compute the intersection between the learned vocabularies.


## Folder Structure
- `data/`: Contains the training, validation and testing data for the language models.
- `src/`:
  - `Normalizer.py`: Implements text normalization techniques.
  - `Tokenizer.py`: Handles tokenization and N-gram generation.
  - `NgramModel.py`: Implements the N-gram language models (Add-alpha, interpolation).
  - `LIdentify.py`: Provides the language identification functionality.
  - `BytePairEncoding.py`: Includes the code for creating a byte pair encoding.
  - `utils.py`: Contains various utility functions.
- `report/`: Holds the LaTeX source code for the project report.
- `main.ipynb`: The main Jupyter notebook that serves as the driver code for the project.

## Usage
1. Ensure you have the necessary data files in the `data/` directory (as .txt).
2. Run the `main.ipynb` Jupyter notebook to execute the language detection and similarity analysis.
3. The project report can be compiled from the LaTeX source code in the `report/` directory.

**Note**: The NGram model can be used for a higher order N-Gram model as well.

## Dependencies
- Python 3.x
- NumPy
- Seaborn
- matplotlib
- Jupyter Notebook

## References
* **Herman Kamper**, NLP817, https://www.kamperh.com/nlp817/, Stellenbosch University
* **Daniel Jurafsky**, Speech and Language Processing, 3rd Edition, Stanford University
* **Compare Language** http://www.elinguistics.net/Compare_Languages.aspx

