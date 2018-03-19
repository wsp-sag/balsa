from io import FileIO
from typing import Union, Optional, Dict, Any
from six import string_types

try:
    from pathlib import Path
    file_types = Union[string_types, Path, FileIO]
except ImportError:
    Path = None
    file_types = Union[string_types, FileIO]


class ConfigValue:
    pass


class Config:

    _contents: dict
    _name: str
    _parent: Optional[Config]
    _file = Optional[str]

    def name(self) -> string_types: pass

    def parent(self) -> Optional['Config']: pass

    def namespace(self) -> string_types: pass

    def as_dict(self, key_type: type=None, value_type: type=None) -> dict: pass

    def serialize(self) -> string_types: pass

    def to_file(self, fp: file_types): pass

    @classmethod
    def from_file(cls, fp: file_types) -> 'Config': pass

    @classmethod
    def from_string(cls, s: string_types, file_name: string_types, root_name: string_types) -> 'Config': pass

    @staticmethod
    def from_dict(dict_: Dict[string_types, Any], file_name: string_types, root_name: string_types) -> 'Config': pass
