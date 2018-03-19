from __future__ import division, absolute_import, print_function, unicode_literals

import json
import os
from collections import OrderedDict
from six import iteritems, StringIO
import re

try:
    from pathlib import Path
    PATHLIB_LOADED = True
except ImportError:
    PATHLIB_LOADED = False

from balsa.utils import is_identifier, open_file


class ConfigParseError(IOError):
    pass


class ConfigSpecificationError(AttributeError):
    pass


class ConfigTypeError(ValueError):
    pass


class ConfigValue(object):
    """
    Wraps the value of a Config attribute to facilitate type-checking and pretty error messages.

    Attributes:
        value: Get or set the underlying value of the wrapper.
        namespace: The dot-separated namespace of this attribute within the full Config.

    """

    def __init__(self, value, name, owner=None):
        self.value = value
        self._name = str(name)
        self._owner = owner

    def __str__(self): return str(self.value)

    def __repr__(self): return "ConfigValue(%r)" % self.value

    @property
    def namespace(self):
        if self._owner is not None:
            return self._owner.namespace + '.' + self._name
        return self._name

    def as_type(self, type_):
        """
        Attempts to cast the value to a specified type.

        Args:
            type_ (type): The type (e.g. int, float, etc.) to try to cast

        Returns: The value cast as type

        Raises: ConfigTypeError if the casting could not be performed.

        """

        try:
            return type_(self.value)
        except (ValueError, TypeError):

            message = "Attribute <{}> = '{}' could not be converted to {}".format(
                self.namespace, self.value, type_
            )
            raise ConfigTypeError(message)

    def as_bool(self): return self.as_type(bool)

    def as_int(self): return self.as_type(int)

    def as_float(self): return self.as_type(float)

    def as_str(self): return self.as_type(str)

    def as_list(self, sub_type=None):
        """
        Converts the value to a list.

        Args:
            sub_type (type): Optional. Specifies the expected contiguous (uniform) type of the list to convert to.

        Returns:
            list: The value, as a list

        """
        if sub_type is None:
            return self.as_type(list)

        return [
            item.as_type(sub_type)
            for item in self.as_type(list)
        ]

    if PATHLIB_LOADED:
        def as_path(self, parent=None):
            if parent is not None: return Path(parent) / Path(self.as_str())
            return Path(self.as_str())

    def as_set(self, sub_type=None):
        """
        Converts the value to a set.

        Args:
            sub_type (type): Optional. Specifies the expected contiguous (uniform) type of the set to convert to.

        Returns:
            set: The value, as a set

        """
        if sub_type is None: return self.as_type(set)

        return {
            item.as_type(sub_type)
            for item in self.as_type(set)
        }

    def serialize(self):
        if isinstance(self.value, list):
            return [x.serialize() for x in self.value]
        return self.value


class Config(object):
    """
    Represents a model configuration, usually stored in JSON format with the order of items preserved and comments
    (beginning with '//') stripped out. Keys in the JSON file which conform to Python variable names (e.g.
    "my_attribute" but not "My Attribute") become *attributes* of the Config object (e.g. instance.my_attribute).

    Value attributes (e.g. `value` in `{"key": value}`) are stored as ConfigValue objects to facilitate type conversion
    and checking. So to access the raw value, write "instance.my_attribute.value" or, to convert it to a specified type,
    write "instance.my_attribute.as_bool()".

    This all facilitates "pretty" error message generation, to provide the end-user with as much information about the
    source of an error as these are common when specifying a model.

    A Config can be constructed from three static methods:
        - from_file() to construct from a JSON file on-disk
        - from_string() to construct from a JSON-formatted string in-memory
        - from_dict() to construct from a dictionary in-memory

    Notes:
        - Config implements __contains__ for testing if a name is 'in' the set of attributes.
        - To use __getitem__, __setitem__ (like a Dictionary), use the `as_dict()` method to convert to a dictionary
            representation. This also exposes dictionary iteration methods.

    Attributes:
        name: Short name of each part of the config. For non-root Configs, this will be the name of the attribute used
            to access this Config from the parent.
        parent: Pointer to the parent of non-root Configs.
        namespace: The dot-separated namespace of this part of the full Config.

    """

    def __init__(self, config_dict, name=None, parent=None, file_=None):
        self._contents = {}
        self._name = name
        self._parent = parent
        self._file = file_

        for key, original_value in iteritems(config_dict):
            if isinstance(original_value, dict):
                value = Config(original_value, name=key, parent=self, file_=file_)
            elif isinstance(original_value, (list, set)):
                value_list = []
                for (i, item) in enumerate(original_value):
                    if isinstance(item, dict):
                        value_list.append(Config(item, name=key + "[%s]" % i, parent=self, file_=file_))
                    else:
                        value_list.append(ConfigValue(item, key + "[%s]" % i, owner=self))
                value = ConfigValue(value_list, key, owner=self)
            elif original_value is None:
                value = None
            else:
                value = ConfigValue(original_value, key, owner=self)

            if is_identifier(key):
                try:
                    setattr(self, key, value)
                except AttributeError:
                    print("WARNING: Config key '%s' conflicts with reserved properties" % key)
            self._contents[key] = value

    @property
    def name(self): return self._name

    @property
    def parent(self): return self._parent

    @property
    def namespace(self):
        name = self._name if self._name is not None else '<unnamed>'
        if self._parent is None:
            return name
        return '.'.join([self._parent.namespace, name])

    def __str__(self):
        if self._parent is None:
            return "Config @%s" % self._file

        return "Config(%s) @%s" % (self.namespace, self._file)

    def __getattr__(self, item):
        raise ConfigSpecificationError("Item '%s' is missing from config <%s>" % (item, self.namespace))

    def __contains__(self, item): return item in self._contents

    def __getitem__(self, item):
        if item not in self:
            raise ConfigSpecificationError("Item '%s' is missing from config <%s>" % (item, self.namespace))
        return self._contents[item]

    def as_dict(self, key_type=None, value_type=None):
        """
        Converts this entry to a primitive dictionary, using specified types for the keys and values.

        Args:
            key_type (type): The type to which the keys will be cast, or None to ignore casting.
            value_type (type): The type to which the values will be cast, or None to ignore casting.

        Returns: dict

        """

        if key_type is None and value_type is None:
            return self._contents.copy()

        def any_type(val): return val

        if key_type is None: key_type = any_type
        if value_type is None: value_type = any_type

        retval = OrderedDict()
        for key, val in iteritems(self._contents):
            try:
                key = key_type(key)
            except ValueError:
                message = "Key <{}> = '{}' could not be converted to {}".format(
                    self.namespace, key, key_type
                )
                raise ConfigTypeError(message)

            try:
                val = val.as_type(value_type)
            except ValueError:
                message = "Value <{}.{}> = '{}' could not be converted to {}".format(
                    self.namespace, key, val, key_type
                )
                raise ConfigTypeError(message)
            retval[key] = val
        return retval

    def serialize(self):
        """Recursively converts the Config back to primitive dictionaries"""
        child_dict = OrderedDict()
        for attr, item in iteritems(self._contents):
            child_dict[attr] = item.serialize()
        return child_dict

    def to_file(self, fp):
        """
        Writes the Config to a JSON file.

        Args:
            fp (str): File path to the output files

        """
        dict_ = self.serialize()
        with open_file(fp, mode='w') as writer:
            json.dump(dict_, writer, indent=2)

    @classmethod
    def from_file(cls, fp):
        """
        Reads a Config from a JSON file. Comments beginning with '//' are ignored.

        Args:
            fp (str): The path to the JSON file

        Returns:
            Config: The Config object representing the JSON file.

        Raises:
            ConfigParseError if there's a problem parsing the JSON file

        """
        with open_file(fp, mode='r') as reader:
            try:
                dict_ = json.loads(cls._parse_comments(reader), object_pairs_hook=OrderedDict)
            except ValueError as ve:
                # If there's an error reading the JSON file, re-raise it as a ConfigParseError for clarity
                raise ConfigParseError(str(ve))

            root_name = os.path.splitext(os.path.basename(fp))[0]
            return Config(dict_, name=root_name, file_=fp)

    @classmethod
    def from_string(cls, s, file_name='<from_str>', root_name='<root>'):
        """
        Reads a Config from a JSON file as a string. Comments beginning with '//' are ignored.

        Args:
            s (str): The string containing the Config data, in JSON format.
            file_name (str): Optional 'file' name for display purposes.
            root_name (str): Optional root name for display purposes.

        Returns:
            Config: The Config object representing the JSON file.

        Raises:
            ConfigParseError if there's a problem parsing the JSON file

        """
        sio = StringIO(s)
        try:
            dict_ = json.loads(cls._parse_comments(sio), object_pairs_hook=OrderedDict)
        except ValueError as ve:
            raise ConfigParseError(str(ve))

        return Config(dict_, name=root_name, file_=file_name)

    @staticmethod
    def from_dict(dict_, file_name='<from_dict>', root_name='<root>'):
        """
        Converts a raw dictionary to a Config object.

        Args:
            dict_ (dict): The
            file_name:
            root_name:

        Returns:

        """
        return Config(dict_, name=root_name, file_=file_name)

    @staticmethod
    def _parse_comments(reader):
        """Removes comments beginning with '//' from the stream"""
        regex = r'\s*(#|\/{2}).*$'
        regex_inline = r'(:?(?:\s)*([A-Za-z\d\.{}]*)|((?<=\").*\"),?)(?:\s)*(((#|(\/{2})).*)|)$'

        pipe = []
        for line in reader:
            if re.search(regex, line):
                if re.search(r'^' + regex, line, re.IGNORECASE): continue
                elif re.search(regex_inline, line):
                    pipe.append(re.sub(regex_inline, r'\1', line))
            else:
                pipe.append(line)
        return "\n".join(pipe)
