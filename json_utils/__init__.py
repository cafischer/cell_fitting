import base64
import json
import numpy as np

__author__ = 'caro'


def save_as_json(file_dir, d, contains_array=False):
    """
    Saves the input variable as .json file.

    :param file_dir: Directory for saving.
    :type file_dir: str
    :param d: Variable ot be saved. Can be of any .json encodable format. If it contains a ndarray,
    turn :param contains_array on.
    :type d: dict, list
    :param contains_array: Set to True, when :param d contains a ndarray.
    :type contains_array: bool
    """
    if '.json' not in file_dir:
            file_dir += '.json'
    fw = open(file_dir, 'w')
    if contains_array:
        json.dump(d, fw, indent=4, cls=NumpyEncoder)
    else:
        json.dump(d, fw, indent=4)


def load_json(file_dir, contains_array=False):
    """
    Loads a variable from a .json file.

    :param file_dir: Directory where variable was saved.
    :type file_dir: str
    :param contains_array: Set to True, when :param d contained a ndarray.
    :type contains_array: bool
    :return: Variable loaded from .json file.
    :rtype: dict, list
    """
    fr = open(file_dir, 'r')
    if contains_array:
        d = json.load(fr, object_hook=json_numpy_obj_hook)
    else:
        d = json.load(fr)
    return d


def json_numpy_obj_hook(dct):
    """Decodes a previously encoded numpy ndarray with proper shape and dtype.

    :param dct: Json encoded ndarray.
    :type dct: dict
    :return: Variable from .json file.
    :rtype: ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        """If input object is an ndarray it will be converted into a dict
        holding dtype, shape and the data, base64 encoded.
        """
        if isinstance(obj, np.ndarray):
            if obj.flags['C_CONTIGUOUS']:
                obj_data = obj.data
            else:
                cont_obj = np.ascontiguousarray(obj)
                assert(cont_obj.flags['C_CONTIGUOUS'])
                obj_data = cont_obj.data
            data_b64 = base64.b64encode(obj_data)
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder(self, obj)