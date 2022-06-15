![Screenshots](https://github.com/aivarovsky/3fa-authenticator-app/blob/main/images/screenshots.png)

# 3FA Authenticator App

## Overview

3FA Authenticator App introduces a third factor to the conventional two-factor authentication (2FA) mechanism by incorporating a handwritten signature authenticity check.

## How it works

3FA Authenticator App verifies three factors: knowledge, inherence and possession.

The first and second factors are managed by the authentication system, which verifies the user's password and handwritten signature. The handwritten signature is verified using a siamese neural network model. The third factor is handled by the one-time password generator. The generator is only accessible after successful authentication.

> [!CAUTION]
> 3FA Authenticator App creates `anchor` and `dict` files on user registration:
> - The `dict` file stores encrypted information about the user's account names and secret keys.
> - The `anchor` file stores encrypted information about the user's handwritten signature.
> 
> The user's password is used for encryption key generation. Forgetting it or deleting these files will result in irreversible data loss.

## Siamese Neural Network Model Architecture

![Architecture](https://github.com/aivarovsky/3fa-authenticator-app/blob/main/images/architecture.png)

## Modules

| Module | Description |
|---|---|
| [main.py](https://github.com/aivarovsky/3fa-authenticator-app/blob/main/main.py) | Main script that runs the app. |
| [datacrypt.py](https://github.com/aivarovsky/3fa-authenticator-app/blob/main/datacrypt.py) | Functions for encrypting and decrypting user data. |
| [preprocess.py](https://github.com/aivarovsky/3fa-authenticator-app/blob/main/preprocess.py) | Functions for preprocessing signatures for the model. |
| [train.py](https://github.com/aivarovsky/3fa-authenticator-app/blob/main/train.py) | Functions used for creating, training, validating, and testing the model. |
| [siamese](https://github.com/aivarovsky/3fa-authenticator-app/blob/main/siamese) | Model data. |

## Setup

1. Clone this repository:

    ```bash
    git clone https://github.com/aivarovsky/3fa-authenticator-app
    ```

2. Navigate to the project directory:

    ```bash
    cd 3fa-authenticator-app
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run `main.py`:

    ```bash
    python main.py
    ```

## References

- [Cedar Signature Dataset](https://paperswithcode.com/dataset/cedar-signature) ([Download](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset))

## License

This project is licensed under the [**MIT License**](https://github.com/aivarovsky/3fa-authenticator-app/blob/main/LICENSE).