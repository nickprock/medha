import logging
import os
import pytest
from medha.logging import setup_logging, LIBRARY_LOGGER_NAME

@pytest.fixture(autouse=True)
def cleanup_handlers():
    """Ensure the logger is cleared before and after each test."""
    logger = logging.getLogger(LIBRARY_LOGGER_NAME)
    logger.handlers.clear()
    yield
    logger.handlers.clear()

def test_setup_logging_defaults():
    """Test that default setup creates a StreamHandler with INFO level."""
    logger = setup_logging()
    
    assert logger.name == LIBRARY_LOGGER_NAME
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert logger.propagate is False

def test_setup_logging_custom_level():
    """Test that passing a custom level updates the logger and handler."""
    logger = setup_logging(level="DEBUG")
    
    assert logger.level == logging.DEBUG
    # The console handler should match the main level by default
    assert logger.handlers[0].level == logging.DEBUG

def test_console_level_override():
    """Test that console_level can differ from the main logger level."""
    logger = setup_logging(level="INFO", console_level="ERROR")
    
    assert logger.level == logging.INFO
    assert logger.handlers[0].level == logging.ERROR

def test_file_handler_creation(tmp_path):
    """Test that a file handler is added when a log_file path is provided."""
    log_file = tmp_path / "test.log"
    logger = setup_logging(level="INFO", log_file=str(log_file))
    
    # Should have two handlers: Console and File
    assert len(logger.handlers) == 2
    
    file_handler = next(h for h in logger.handlers if isinstance(h, logging.FileHandler))
    assert file_handler.baseFilename == str(log_file)
    
    # Verify logging actually writes to the file
    logger.info("Test log message")
    assert log_file.exists()
    assert "Test log message" in log_file.read_text()

def test_handler_deduplication():
    """Ensure calling setup_logging multiple times clears old handlers."""
    setup_logging()
    setup_logging()
    setup_logging()
    
    logger = logging.getLogger(LIBRARY_LOGGER_NAME)
    assert len(logger.handlers) == 1

def test_invalid_level_fallback():
    """Ensure it falls back to INFO if an invalid string is passed."""
    logger = setup_logging(level="NOT_A_LEVEL")
    assert logger.level == logging.INFO