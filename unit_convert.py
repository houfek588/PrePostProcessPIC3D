

# l1 = [0, 1, 2, 3, 4, 5]

def rescale_list(input_list, scale):
    return [n * scale for n in input_list]


def rescale_list_of_lists(lst, multiplier):
    """
    Multiplies every element in a list of lists by a specified multiplier.

    Args:
        lst (list of lists): The input list of lists containing numeric elements.
        multiplier (float or int): The number to multiply each element by.

    Returns:
        list of lists: A new list of lists with each element multiplied by the multiplier.
    """
    return [[element * multiplier for element in sublist] for sublist in lst]

# l1 = [[1, 2, 3],
#       [4, 5, 6],
#       [7, 8, 9]]
# print(rescale_list_of_lists(l1,3))