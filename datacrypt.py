import json
import base64
import numpy as np
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


def KDF(password):
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=None)
    return Fernet(base64.urlsafe_b64encode(hkdf.derive(password.encode())))


def encrypt(data, password):
    if isinstance(data, np.ndarray):
        data = data.tobytes()
    else:
        data = json.dumps(data).encode()
    return KDF(password).encrypt(data)


def decrypt(data, password, dims=None, dtype=None):
    try:
        decrypted_data = KDF(password).decrypt(data)
    except InvalidToken:
        return -1

    if dims is None and dtype is None:
        decrypted_data = json.loads(decrypted_data.decode())
    else:
        decrypted_data = np.frombuffer(decrypted_data, dtype=dtype).reshape(dims)
    return decrypted_data
