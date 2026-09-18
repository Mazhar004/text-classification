from datetime import datetime as dt
import argparse
import os
import shutil

# Custom
import paths


# os.path.altsep is None on POSIX, so it is filtered out rather than being
# folded into '' -- an empty string is a substring of everything.
PATH_SEPARATORS = tuple(s for s in (os.path.sep, os.path.altsep) if s)


def safe_folder_name(name):
    """Reject names that would escape the previous_version directory."""
    if name and (any(sep in name for sep in PATH_SEPARATORS)
                 or name in ('.', '..')):
        raise ValueError(
            'Invalid --foldername {!r}: it must be a single directory name '
            'inside {}.'.format(name, paths.VERSION_DIR))
    return name


class DataStore():
    """Back up or restore a dataset's data, config, weights and reports."""

    def __init__(self, dataset, foldername=""):
        self.dataset = dataset
        self.foldername = safe_folder_name(foldername)
        self.time = dt.now().strftime("_%d_%b_%I_%M_%S_%p")

        self.version_root = paths.VERSION_DIR / (
            self.foldername or self.dataset + self.time)
        self.slots = self.slot_pairs()

        if self.foldername == "":
            self.backup()
        else:
            self.restore()

    def slot_pairs(self):
        """(live path, archived path) for each item, CSV file first."""
        dataset = self.dataset
        root = self.version_root

        return [
            (paths.csv_file(dataset), root / 'data' / (dataset + '.csv')),
            (paths.data_dir(dataset), root / 'data' / 'data'),
            (paths.CONFIG_DIR / dataset, root / 'config'),
            (paths.weight_dir(dataset), root / 'weight'),
            (paths.performance_dir(dataset), root / 'result'),
        ]

    @staticmethod
    def copy(source, destination):
        if not source.exists():
            print('Skipping missing {}'.format(source))
            return

        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.rmtree(destination, ignore_errors=True)
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
        print('{} -> {}'.format(source, destination))

    def backup(self):
        if self.version_root.exists():
            raise FileExistsError(
                'Backup {} already exists.'.format(self.version_root))

        for live, archived in self.slots:
            self.copy(live, archived)
        print('Backup written to {}'.format(self.version_root))

    def restore(self):
        if not self.version_root.is_dir():
            raise FileNotFoundError(
                'No backup at {}'.format(self.version_root))

        for live, archived in self.slots:
            # Only the dataset's own CSV is removed -- the previous version of
            # this script deleted the whole csv/ directory, taking every other
            # dataset with it.
            if live.is_dir():
                shutil.rmtree(live, ignore_errors=True)
            elif live.exists():
                live.unlink()
            self.copy(archived, live)
        print('Restored {} from {}'.format(self.dataset, self.version_root))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default='assistant', required=False,
                        type=str, help="Name of the dataset")
    parser.add_argument("--foldername", default="", required=False,
                        type=str, help="For restore type foldername")
    args = parser.parse_args()

    datastore = DataStore(args.dataset, args.foldername)
