from pathlib import Path
import numpy as np


def print_array(array2d):
    array2d = np.array(array2d)
    int_array = np.array(
        [[np.real(complex(f"{i:.2f}")).is_integer()
            and np.imag(complex(f"{i:.2f}")).is_integer()
            for i in row] for row in array2d])
    idx_0 = np.where(int_array)
    top_row = np.arange(len(array2d[0]))
    top_str = [""] + [f"{e:2d}" for e in top_row]

    str_mat = np.array([[f"{e:.2f}" for e in row] for row in array2d])

    str_mat[idx_0] = np.array([int(round(np.real(i))) for i in array2d[idx_0]])
    lens = np.max([list(map(len, col)) for col in zip(*str_mat)], axis=1)
    fmt = "{{:>{}}} :  ".format(2) + "  ".join("{{:>{}}}".format(x + 1) for x in lens)
    table = [fmt.format(i, *row) for i, row in enumerate(str_mat)]

    fmt_top = "{{:>{}}}    ".format(2) + "  ".join("{{:>{}}}".format(x + 1) for x in lens)
    top_table = fmt_top.format(*top_str)
    top_print = "   " + "".join(top_table)[3:]

    print("\n", top_print)
    print("     " + "-" * (len(top_print) - 4))
    print("\n".join(table))


def print_idx(rho):
    idx = np.where(rho != 0)
    print_array(np.vstack((idx, rho[idx])).T)


def remove(remove_path, keep_dir=False):
    if remove_path.is_dir():
        for child in remove_path.iterdir():
            remove(child)
        if not keep_dir:
            remove_path.rmdir()
    else:
        remove_path.unlink()


def save_dict(save_path, *args, suppress_warning=False, remove_dir=False):
    save_path = Path(save_path)
    data_dict = {}
    for a_dict in args:
        data_dict = {**data_dict, **a_dict}

    if save_path.exists():
        if not suppress_warning:
            print(f"Warning: {save_path} exists. Data will be overwritten!")
            input("Press Enter to continue...")
        if remove_dir:
            remove(save_path)
            save_path.mkdir(parents=True)
    else:
        save_path.mkdir(parents=True)

    for key, value in data_dict.items():
        np.save(save_path / f"{key}", value)


def load_dict(load_path):
    load_path = Path(load_path)
    data_dict = {}

    for child in load_path.iterdir():
        value = np.load(child)
        key = child.stem
        data_dict[key] = value

    return data_dict


def find_folder(search_folder, name, print_all=False):
    """Finds all files/folder with {name} in their names

    Iterates through a folder finding all files/folder including {name} in
    their names returning all the files/folders fitting this criteria

    Parameters:
        search_folder (string): the folder in which to search
        name (string): the string to search for
        print_all (bool): whether to print a list of all hits

    Returns (list): a list with the paths to all files/folders with {name} in
    """
    child_list = []
    if list(Path(search_folder).glob("*" + name + "*")):
        child_list.append(list(Path(search_folder).glob("*" + name + "*")))

    def iter_folder(search_folder):
        try:
            for child in Path(search_folder).iterdir():
                if list(child.glob("*" + name + "*")):
                    child_list.append(list(child.glob("*" + name + "*")))
                else:
                    iter_folder(child)
            return None
        except NotADirectoryError:
            return None

    iter_folder(search_folder)
    if print_all:
        for el in child_list:
            for em in el:
                print(em)

    return np.array(child_list).flatten()


def exist_in_folder(search_folder, name):
    if list(find_folder(search_folder, name, print_all=False)):
        return True
    else:
        return False


def print_dict(dict_to_print):
    for key, value in dict_to_print.items():
        print(f"{key:20}: {value}")


def find_next_word(long_string, word):
    for i, piece in enumerate(long_string.split()):
        if word in piece:
            return long_string.split()[i + 1]
    return None


# NOTE: Slow, but works with autograd
def array_assignment(mat, mini_mat, pos):
    mat = np.array(mat)
    mini_mat = np.array(mini_mat)

    if mini_mat.ndim == 0:
        mini_mat = np.array([[mini_mat]])
    elif mini_mat.ndim == 1:
        mini_mat = np.array([mini_mat])

    new_mat = []
    for i, row in enumerate(mat):
        new_row = []
        for j, el in enumerate(row):
            if ((i >= pos[0] and i < pos[0] + mini_mat.shape[0])
                    and (j >= pos[1] and j < pos[1] + mini_mat.shape[1])):
                new_row.append(mini_mat[i - pos[0]][j - pos[1]])
            else:
                new_row.append(mat[i][j])
        new_mat.append(new_row)

    return np.array(new_mat)
