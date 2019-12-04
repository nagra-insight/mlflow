import os
import logging

from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.tracking.context.default_context import _get_main_file
from mlflow.utils.mlflow_tags import MLFLOW_GIT_COMMIT

_logger = logging.getLogger(__name__)


def _get_git_commit(path, clean_check=False):
    try:
        import git
    except ImportError as e:
        _logger.warning(
            "Failed to import Git (the Git executable is probably not on your PATH),"
            " so Git SHA is not available. Error: %s", e)
        return None
    try:
        if os.path.isfile(path):
            path = os.path.dirname(path)
        repo = git.Repo(path, search_parent_directories=True)
        if clean_check and repo.is_dirty(path=path, untracked_files=True):
            _logger.info('Uncommitted changes found in %s, commit sha1 will not be used', path)
            commit = None
        else:
            commit = repo.head.commit.hexsha
        return commit
    except (git.InvalidGitRepositoryError, git.GitCommandNotFound, ValueError, git.NoSuchPathError):
        return None


def _get_source_version():
    main_file = _get_main_file()
    if main_file is not None:
        return _get_git_commit(main_file)
    return None


class GitRunContext(RunContextProvider):

    def __init__(self):
        self._cache = {}

    @property
    def _source_version(self):
        if "source_version" not in self._cache:
            self._cache["source_version"] = _get_source_version()
        return self._cache["source_version"]

    def in_context(self):
        return self._source_version is not None

    def tags(self):
        return {
            MLFLOW_GIT_COMMIT: self._source_version
        }
