# cedar

Model configuration reading, writing, and validation utilities

## `Config` objects

The `Config` class is the main component of this library, and is designed to facilitate working with a slightly-extended JSON configuration format. Its main feature is raising pretty and informative error messages to the end-user, with minimal code required on the programmer's part. 

Keys in the JSON file which conform to Python variable names become *attributes* of the Config object. For example, if you JSON file is
```json
//my_config.json
{
  "top_attribute": true,
  "sub_module_1": {
    "sub_attribute_1": 5,
    "sub_attribute_2": "C:/some_file.txt"
  }
}
```

then you can access `sub_attribute_1` like this:

```python
from cedar import Config

my_config = Config.from_file("my_config.json)
print(my_config.sub_module_1.sub_attribute_1.as_int())
```

If `sub_module_1` doesn't contain an attribute `sub_attribute_1` a pretty error message will be raised:
```
>> ConfigSpecificationError: Item 'sub_attribute_1' is missing from config <my_config.sub_module_1>
```

If it's not of the correct type, a different error is raised:
```
>> ConfigTypeError: Attribute <my_config.sub_module_1.sub_attribute_1> = 'foobar' could not be converted to <type 'int'>
```
