from pathlib import Path


# Will be called from the root, circleci (project) or from a subfolder
PATH_PROJECT = (
    Path('.') if (Path('.').resolve().name == 'Data-And-Model-Drift-Checker'
                  or Path('.').resolve().name == 'project')
    else Path('..')
)

PATH_DATA = PATH_PROJECT / 'datasets'
RANDOM_STATE = 1994
