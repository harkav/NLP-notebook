from src.most_similar import validate_files
import pytest


# got help from chatgpt. I had originally written tests for all the cases, 
# but I used files on my computer. I thought that it would be better to find 
# a way to use temp files, but didn't know how, so chatgpt helped me out. 


def test_validate_files_empty_dir(tmp_path):
    """Should raise NotADirectoryError for a path to a non dir given as dir_path"""
    with pytest.raises(NotADirectoryError):
        validate_files("not_a_dir", str(tmp_path / "input.txt"))


def test_raises_value_error_if_dir_contains_only_empty_files(tmp_path):
    # Create empty corpus files
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "empty1.txt").write_text("")
    (corpus_dir / "empty2.txt").write_text("")

    # Create input file outside corpus dir
    input_file = tmp_path / "input.txt"
    input_file.write_text("non empty")

    with pytest.raises(ValueError):
        validate_files(str(corpus_dir), str(input_file))


def test_raise_filenotfounderror_for_non_existing_inputfile(tmp_path):
    """Should raise FileNotFoundError for faulty input file"""

    # create one valid non-empty file in dir
    (tmp_path / "file1.txt").write_text("content")

    # provide non-existent input file path
    non_existent_input = tmp_path / "no_such_file.txt"

    with pytest.raises(FileNotFoundError):
        validate_files(str(tmp_path), str(non_existent_input))


def test_raise_value_error_for_empty_input_file(tmp_path):
    """Should raise ValueError for empty input file"""

    # create one valid non-empty file in dir
    (tmp_path / "file1.txt").write_text("content")

    # create empty input file
    empty_input = tmp_path / "input.txt"
    empty_input.write_text("")

    with pytest.raises(ValueError):
        validate_files(str(tmp_path), str(empty_input))


def test_accepts_valid_input(tmp_path):
    """Should accept valid input"""

    # create one valid non-empty file in dir
    (tmp_path / "file1.txt").write_text("content")

    # create valid non-empty input file
    input_file = tmp_path / "input.txt"
    input_file.write_text("valid input")

    assert validate_files(str(tmp_path), str(input_file)) is True
