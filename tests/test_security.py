from vitaltrace.security import hash_password, verify_password


def test_hash_and_verify():
    stored = hash_password("segredo")
    assert stored != "segredo"
    assert verify_password("segredo", stored)
    assert not verify_password("outra", stored)


def test_hash_is_salted():
    assert hash_password("x") != hash_password("x")


def test_verify_rejects_malformed_hash():
    assert not verify_password("x", "lixo")
