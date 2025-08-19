# assigns a default value to a dictionary
def dict_default(dict: dict, key: str, default: any):
    if key not in dict:
        dict[key] = default