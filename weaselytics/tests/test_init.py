import weaselytics


class TestInit:
    def test_version_is_defined(self):
        assert weaselytics.__version__ is not None
        assert isinstance(weaselytics.__version__, str)
        assert len(weaselytics.__version__) > 0

    def test_all_names_are_importable(self):
        for name in weaselytics.__all__:
            assert hasattr(weaselytics, name), f"{name} not found in weaselytics"
