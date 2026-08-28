import os


def read_lines(file_name):
    """loads class name from a file"""
    classes_path = os.path.expanduser(file_name)
    with open(classes_path, encoding='utf8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names
