import os
import json


class CustomEncoder(json.JSONEncoder):
    """
    A class representing a custom JSON encoder for serializing objects with custom dictionary representations.
    """

    def default(self, obj):
        """
        Serializes the object to a JSON-compatible format.
        Checks if the object has a 'to_dict', 'as_dict', or 'model_dump' method and calls it to get the dictionary representation.

        Args:
            obj: The object to serialize.

        Returns:
            str: The serialized object as a JSON-compatible format.
        """

        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if hasattr(obj, 'as_dict'):
            return obj.as_dict()
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        return super().default(obj)


def create_directory(dir: str, clear_if_not_empty: bool = False) -> str:
    os.makedirs(dir, exist_ok=True)

    if clear_if_not_empty:
        for file in os.listdir(dir):
            file_path = os.path.join(dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    return dir


def create_json_file(fpath: str, data: any, indent: int = 4) -> None:
    if not os.path.exists(os.path.dirname(fpath)):
        create_directory(os.path.dirname(fpath))

    with open(fpath, 'w') as f:
        json.dump(data, f, indent=indent, cls=CustomEncoder)


def create_text_file(fpath: str, data: str) -> None:
    if not os.path.exists(os.path.dirname(fpath)):
        create_directory(os.path.dirname(fpath))

    with open(fpath, 'w') as f:
        f.write(data)
