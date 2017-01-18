import json
import re
import os
from collections import OrderedDict
from six import iteritems
from keyword import kwlist
try:
    from StringIO import StringIO  # Py 2.x
except ImportError:
    from io import StringIO  # Py 3.x


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

    __slots__ = ['value', '_name', '_owner', 'namespace']

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

        """

        try:
            return type_(self.value)
        except ValueError:

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

        for key, value in iteritems(config_dict):
            if isinstance(value, dict):
                value = Config(value, name=key, parent=self, file_=file_)
            elif isinstance(value, (list, set)):
                value = [
                    (Config(item, name=key + "[%s]" % i, parent=self, file_=file_)
                        if isinstance(item, dict)
                        else ConfigValue(item, key + "[%s]" % i, owner=self))
                    for (i, item) in enumerate(value)
                ]
                value = ConfigValue(value, key, owner=self)
            else:
                value = ConfigValue(value, key, owner=self)

            if self.name_is_pythonic(key):
                setattr(self, key, value)
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

    def as_dict(self):
        """Returns the Config as a dictionary. Sub-dictionaries will still be Configs"""
        return self._contents

    def serialize(self):
        """Recursively converts the Config back to primitive dictionaries"""
        child_dict = OrderedDict()
        for attr, item in self._d.iteritems():
            if isinstance(item, Config):
                child_dict[attr] = item.serialize()
            elif isinstance(item, list):
                child_dict[attr] = [x.serialize() if isinstance(x, Config) else x for x in item]
            else:
                child_dict[attr] = item
        return child_dict

    def to_file(self, fp):
        """
        Writes the Config to a JSON file.

        Args:
            fp (str): File path to the output files

        """
        dict_ = self.serialize()
        with open(fp, 'w') as writer:
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
        with open(fp, 'r') as reader:
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
    def name_is_pythonic(name):
        """
        Tests that the name is a valid Python variable name and does not collide with reserved keywords

        Args:
            name (str): Name to test

        Returns:
            bool: If the name is 'Pythonic'

        """

        special_chars = set(r" .,<>/?;:'|[{]}=+-)(*&^%$#@!`~" + '"')
        regex_chars = set(r"]")
        pyspecchar = list(special_chars - regex_chars)
        escaped_chars = ["\%s" % c for c in regex_chars]
        insertion = ''.join(pyspecchar + escaped_chars)
        unpythonic_regex = r"^\d|[%s\s]+" % insertion

        return not re.match(unpythonic_regex, name) or name in kwlist

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
