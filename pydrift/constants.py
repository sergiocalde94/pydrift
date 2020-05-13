from pathlib import Path


# PATH_PROJECT will be called from the root of the project or from a subfolder
PATH_PROJECT = (
    Path('.') if Path('.').resolve().name == 'Data-And-Model-Drift-Checker'
    else Path('..')
)

PATH_DATA = PATH_PROJECT / 'data'
RANDOM_STATE = 1994
