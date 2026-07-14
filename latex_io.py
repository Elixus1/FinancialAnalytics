import numpy as np


def format_matrix(mat, decimals=3, latex=True):
    """
    Formatiert eine beliebige NumPy-Matrix entweder als LaTeX pmatrix
    oder als normales Textformat.

    Parameters
    ----------
    mat : np.ndarray
        Eingabematrix.
    decimals : int
        Anzahl der Nachkommastellen.
    latex : bool
        True  -> LaTeX pmatrix
        False -> normales Textformat

    Returns
    -------
    str : formatierte Matrix als String
    """
    mat = np.atleast_2d(np.asarray(mat))

    fmt = f"{{:.{decimals}f}}"

    rows = [" & ".join(fmt.format(v) for v in row) for row in mat]

    if latex:
        body = r" \\" + "\n"
        return "\\begin{pmatrix}\n" + body.join(rows) + "\n\\end{pmatrix}"
    else:
        return "\n".join(rows)
