import pickle
import base64
from sklearn.linear_model import Ridge

test_object = Ridge()

# pickle the object
pickled_bytes = pickle.dumps(test_object)

# Encode with base64
# This should be appended to the file.
# pickled_bytes_as_string = base64.b64encode(pickled_bytes)
# print(pickled_bytes_as_string)
pickled_bytes_as_string = b"gASVwQAAAAAAAACMG3NrbGVhcm4ubGluZWFyX21vZGVsLl9yaWRnZZSMBVJpZGdllJOUKYGUfZQojAVhbHBoYZRHP/AAAAAAAACMDWZpdF9pbnRlcmNlcHSUiIwJbm9ybWFsaXpllImMBmNvcHlfWJSIjAhtYXhfaXRlcpROjAN0b2yURz9QYk3S8an8jAZzb2x2ZXKUjARhdXRvlIwMcmFuZG9tX3N0YXRllE6MEF9za2xlYXJuX3ZlcnNpb26UjAYwLjIzLjKUdWIu"

# Decode with base64
decoded_pickled_bytes = base64.b64decode(pickled_bytes_as_string)

# Unpickle object
unpickled_object = pickle.loads(decoded_pickled_bytes)

print(unpickled_object)
