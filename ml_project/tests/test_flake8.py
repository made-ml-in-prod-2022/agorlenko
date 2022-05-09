from flake8.main import application


def test_flake8(code_paths):
    app = application.Application()
    app.run(code_paths)
    assert app.result_count == 0, 'Flake8 found code style errors or warnings'
