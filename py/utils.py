import json
import os
import re
import folder_paths

class AnyType(str):
  """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

  def __ne__(self, __value: object) -> bool:
    return False

class FlexibleOptionalInputType(dict):
  """A special class to make flexible nodes that pass data to our python handlers.

  Enables both flexible/dynamic input types (like for Any Switch) or a dynamic number of inputs
  (like for Any Switch, Context Switch, Context Merge, Power Lora Loader, etc).

  Note, for ComfyUI, all that's needed is the `__contains__` override below, which tells ComfyUI
  that our node will handle the input, regardless of what it is.

  However, with https://github.com/comfyanonymous/ComfyUI/pull/2666 a large change would occur
  requiring more details on the input itself. There, we need to return a list/tuple where the first
  item is the type. This can be a real type, or use the AnyType for additional flexibility.

  This should be forwards compatible unless more changes occur in the PR.
  """
  def __init__(self, type):
    self.type = type

  def __getitem__(self, key):
    return (self.type, )

  def __contains__(self, key):
    return True


any_type = AnyType("*")


def is_dict_value_falsy(data: dict, dict_key: str):
  """ Checks if a dict value is falsy."""
  val = get_dict_value(data, dict_key)
  return not val


def get_dict_value(data: dict, dict_key: str, default=None):
  """ Gets a deeply nested value given a dot-delimited key."""
  keys = dict_key.split('.')
  key = keys.pop(0) if len(keys) > 0 else None
  found = data[key] if key in data else None
  if found is not None and len(keys) > 0:
    return get_dict_value(found, '.'.join(keys), default)
  return found if found is not None else default


def set_dict_value(data: dict, dict_key: str, value, create_missing_objects=True):
  """ Sets a deeply nested value given a dot-delimited key."""
  keys = dict_key.split('.')
  key = keys.pop(0) if len(keys) > 0 else None
  if key not in data:
    if create_missing_objects == False:
      return
    data[key] = {}
  if len(keys) == 0:
    data[key] = value
  else:
    set_dict_value(data[key], '.'.join(keys), value, create_missing_objects)

  return data


def dict_has_key(data: dict, dict_key):
  """ Checks if a dict has a deeply nested dot-delimited key."""
  keys = dict_key.split('.')
  key = keys.pop(0) if len(keys) > 0 else None
  if key is None or key not in data:
    return False
  if len(keys) == 0:
    return True
  return dict_has_key(data[key], '.'.join(keys))


def load_json_file(file: str, default=None):
  """Reads a json file and returns the json dict, stripping out "//" comments first."""
  if path_exists(file):
    with open(file, 'r', encoding='UTF-8') as file:
      config = file.read()
      try:
        return json.loads(config)
      except json.decoder.JSONDecodeError:
        try:
          config = re.sub(r"^\s*//\s.*", "", config, flags=re.MULTILINE)
          return json.loads(config)
        except json.decoder.JSONDecodeError:
          try:
            config = re.sub(r"(?:^|\s)//.*", "", config, flags=re.MULTILINE)
            return json.loads(config)
          except json.decoder.JSONDecodeError:
            pass
  return default


def save_json_file(file_path: str, data: dict):
  """Saves a json file."""
  os.makedirs(os.path.dirname(file_path), exist_ok=True)
  with open(file_path, 'w+', encoding='UTF-8') as file:
    json.dump(data, file, sort_keys=False, indent=2, separators=(",", ": "))

def load_json_from_file(file_path):
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}")
        raise

def save_dict_to_json(data_dict, file_path):
    try:
        with open(file_path, 'w') as json_file:
            json.dump(data_dict, json_file, indent=4)
            print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving JSON to file: {e}")

def show_list(list_input):
    i = 0
    output = ""
    for debug in list_input:
        output += f"{debug},"
        i+=1
    return output

def load_and_save_tags(lora_name, force_fetch):
    json_tags_path = "./loras_tags.json"
    lora_tags = load_json_from_file(json_tags_path)
    output_tags = lora_tags.get(lora_name, None) if lora_tags is not None else None
    if output_tags is not None:
        output_tags_list = output_tags
    else:
        output_tags_list = []

    lora_path = folder_paths.get_full_path("loras", lora_name)
    if lora_tags is None or force_fetch or output_tags is None: # search on civitai only if no local cache or forced
        print("[Lora-Auto-Trigger] calculating lora hash")
        LORAsha256 = calculate_sha256(lora_path)
        print("[Lora-Auto-Trigger] requesting infos")
        model_info = get_model_version_info(LORAsha256)
        if model_info is not None:
            if "trainedWords" in model_info:
                print("[Lora-Auto-Trigger] tags found!")
                if lora_tags is None:
                    lora_tags = {}
                lora_tags[lora_name] = model_info["trainedWords"]
                save_dict_to_json(lora_tags, json_tags_path)
                output_tags_list = model_info["trainedWords"]
        else:
            print("[Lora-Auto-Trigger] No informations found.")
            if lora_tags is None:
                    lora_tags = {}
            lora_tags[lora_name] = []
            save_dict_to_json(lora_tags,json_tags_path)

    return output_tags_list

def get_model_version_info(hash_value):
    api_url = f"https://civitai.com/api/v1/model-versions/by-hash/{hash_value}"
    response = requests.get(api_url)
    
    if response.status_code == 200:
        return response.json()
    else:
        return None

def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def path_exists(path):
  """Checks if a path exists, accepting None type."""
  if path is not None:
    return os.path.exists(path)
  return False


class ByPassTypeTuple(tuple):
  """A special class that will return additional "AnyType" strings beyond defined values.
  Credit to Trung0246
  """

  def __getitem__(self, index):
    if index > len(self) - 1:
      return AnyType("*")
    return super().__getitem__(index)
