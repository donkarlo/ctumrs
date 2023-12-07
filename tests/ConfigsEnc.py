import json
import base64
from cryptography.fernet import Fernet

def generate_key():
    return Fernet.generate_key()

def encrypt_data(data, key):
    cipher = Fernet(key)
    serialized_data = json.dumps(data).encode('utf-8')
    encrypted_data = cipher.encrypt(serialized_data)
    encoded_data = base64.urlsafe_b64encode(encrypted_data).decode('utf-8')
    return encoded_data

def decrypt_data(encoded_data, key):
    cipher = Fernet(key)
    decrypted_data = cipher.decrypt(base64.urlsafe_b64decode(encoded_data.encode('utf-8')))
    return json.loads(decrypted_data.decode('utf-8'))

# Example usage
original_dict = {
    'name': 'John',
    'age': 30,
    'address': {
        'city': 'Example City',
        'zip': '12345'
    }
}

# Generate a key (keep it secure, you'll need it for decryption)
encryption_key = generate_key()

# Encrypt the dictionary
encrypted_string = encrypt_data(original_dict, encryption_key)
print("Encrypted String:", encrypted_string)

# Decrypt the string back to the dictionary
decrypted_dict = decrypt_data(encrypted_string, encryption_key)
print("Decrypted Dictionary:", decrypted_dict)