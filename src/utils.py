from typing import Dict, Tuple, List


def sort_dict(dictionnary: Dict, key: str = "val") -> Dict:
    """
    Sorts a dictionary by key or value.
    Args:
        dictionnary: dictionary to be sorted
        key: the elemnt to use in sorting the dictionary

    Returns:
        (dict): sorted dictionary
    """
    if key == "key":
        return dict(sorted(dictionnary.items(), key=lambda items: items[0], reverse=True))
    else:
        return dict(sorted(dictionnary.items(), key=lambda items: items[1], reverse=True))


def read_test(file_name: str) -> Tuple[List[str], List[str]]:
    """
    Reads test data from file. and splits it contents into two lists
    containing the label and test data.
    Args:
        file_name: path to the test data file
    Returns:
        (List[str], List[str]): test data and its labels
    """
    y, x = [], []
    with open(file_name) as corpus:
        for line in corpus:
            y.append(line[:2].strip())
            x.append(line[2:].strip())
    return (x, y)
