import requests
import uuid
import base64


def randomString(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4())
    random = random.upper()
    random = random.replace("-", "")
    return random[0:string_length]


data = requests.post("https://accounts.spotify.com/api/token", headers={'Content-Type': 'application/x-www-form-urlencoded', 'Authorization': 'Basic NmRkMjNmNDJhNmJkNGEyMzlmNWY3ZmYzZjM0NTU4ZTk6OTlhMGI1YWY4NGFlNDQwYWJhMWY5ZDc4ZDkxYzBhNWM='}, data={
                     "grant_type": "authorization_code", "code": "AQDhGLJFHvy4yZcnYjmgwa7XSKMZupV-JuA93lnyUq-CJLeYVAwubonB52zcCdL2CY-X8vqKqV9xEFUhbwPYvn670c8u--E8_2xaeTbkTbo7rV-44pgZVEvSW0CK1Sb2XNBzASZAMI1yl-sUiHp9V8GTHUkxx8YQ1KH2PuuS560djU8", "redirect_uri": "https://scaredgrippingcalculators.ghelanibhavin.repl.co/auth/callback"})
