import nmr_fido

def test_import_package():
    try:
        import nmr_fido
    except ImportError as e:
        assert False, f"Package import failed: {e}"
    else:
        assert True


def test_version_exists():
    import nmr_fido
    assert hasattr(nmr_fido, "__version__"), "Package does not define __version__"


def test_main_function_exists():
    import nmr_fido
    assert hasattr(nmr_fido, "main"), "Package does not define a 'main' function"