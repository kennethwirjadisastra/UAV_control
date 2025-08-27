# assigns a default value to a dictionary
def add_default_arg(dict: dict, key: str, default: any):
    if key not in dict:
        dict[key] = default
    return dict[key]

def add_arg_with_default(dict: dict, key: str, arg: any, default: any):
    if arg is not None:
        dict[key] = arg
    else:
        dict[key] = default
    return dict[key]