# app/utils/crypto.py
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5
import base64

# 生成 2048 位密钥对
_key = RSA.generate(2048)
PRIVATE_KEY = _key.export_key()
PUBLIC_KEY = _key.publickey().export_key().decode()

def decrypt_password(encrypted_password: str) -> str:
    """使用私钥解密前端传来的 Base64 密文"""
    try:
        # 1. 解码 Base64
        encrypted_bytes = base64.b64decode(encrypted_password)
        # 2. 初始化解密器
        cipher = PKCS1_v1_5.new(RSA.import_key(PRIVATE_KEY))
        # 3. 执行解密，sentinel 是解密失败时的占位符
        sentinel = b"DECRYPT_FAILED"
        decrypted = cipher.decrypt(encrypted_bytes, sentinel)

        if decrypted == sentinel:
            raise ValueError("RSA 解密失败，请检查密钥是否匹配")

        return decrypted.decode('utf-8')
    except Exception as e:
        raise ValueError(f"密码解密异常: {str(e)}")