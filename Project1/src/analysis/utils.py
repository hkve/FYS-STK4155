import pathlib as pl
import seaborn as sns

def make_figs_path(filename):
    cur_path = pl.Path(__file__)
    root_path = cur_path

    while root_path.name != "FYS-STK4155":
        root_path = root_path.parent

    figs_path = root_path / pl.Path("Project1/tex/figs")

    if not figs_path.exists():
        return None
    if not filename.endswith(".pdf"):
        filename += ".pdf"

    figs_path /= filename

    return str(figs_path)

colors = [
    sns.color_palette('husl')[-1],
    sns.color_palette('husl')[-3],
    sns.color_palette('husl')[-2],
    sns.color_palette('husl', 8)[0],
]