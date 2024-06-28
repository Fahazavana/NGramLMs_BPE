def sort_dict(dictionnary, key="val"):
    if key == "key":
        return dict(
            sorted(dictionnary.items(), key=lambda items: items[0], reverse=True)
        )
    else:
        return dict(
            sorted(dictionnary.items(), key=lambda items: items[1], reverse=True)
        )