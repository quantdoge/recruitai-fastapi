import hashlib

def hash_string(input_string):
    """
    Hashes a string input using SHA-256 algorithm.

    Args:
        input_string (str): The string to be hashed

    Returns:
        str: The hexadecimal digest of the hashed string
    """
    if not isinstance(input_string, str):
        input_string = str(input_string)

    # Create a hash object using SHA-256 algorithm
    sha256_hash = hashlib.sha256()

    # Update the hash object with the bytes of the input string
    sha256_hash.update(input_string.encode('utf-8'))

    # Return the hexadecimal digest of the hashed input
    return sha256_hash.hexdigest()


# test_01= hash_string("uat-password-135246!@$")
# print(test_01)  # Example usage, should print the SHA-256 hash of "
