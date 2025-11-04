from dataclasses import dataclass

@dataclass
class Dataset:
    path: str = "keatone/TinierStories"
    split: str = "train"
    name: str = None
    data_dir: str = None
    data_files: str = None

@dataclass
class Data:
    dataset: Dataset = Dataset()
    column: str = "text"
    tok_bs: int = 10000
    pack_bs: int = 100000
    skip_cache: bool = False
