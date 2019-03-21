from io import FileIO
from typing import Union, Optional, Dict, Any, List
from six import string_types

try:
    from pathlib import Path
    PATHLIB_LOADED = True
    file_types = Union[string_types, Path, FileIO]
except ImportError:
    Path = None
    PATHLIB_LOADED = False
    file_types = Union[string_types, FileIO]


class ConfigParseError(IOError):
    pass


class ConfigSpecificationError(AttributeError):
    pass


class ConfigTypeError(ValueError):
    pass


class ConfigValue:

    value: Union[str, List[Union[str, ConfigValue]]]
    _name: str
    _owner: Config

    def namespace(self) -> str:
        pass

    def as_type(self, type_): pass

    def as_bool(self) -> bool: pass

    def as_int(self) -> int: pass

    def as_float(self) -> float: pass

    def as_str(self) -> str: pass

    def as_list(self, sub_type=None) -> list: pass

    if PATHLIB_LOADED:
        def as_path(self, parent: Optional[Path]=None) -> Path:
            pass

    def as_set(self, sub_type=None) -> set: pass

    def serialize(self) -> Union[list, Any]: pass


class Config:

    _contents: dict
    _name: str
    _parent = Optional['Config']
    _file = file_types

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

    @staticmethod
    def _parse_comments(reader): pass
