import glob


def get_filepaths(root):
    """

    :param root:
    :return:
    """
    filepaths = []

    for filepath in glob.glob(root, recursive=True):
        filepaths.append(filepath)

    return filepaths
