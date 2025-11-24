# Test Suite Documentation

## Overview

This test suite provides comprehensive coverage of the Monetary Policy RL project, including unit tests, integration tests, and end-to-end workflow tests.

## Test Structure
```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_environment.py      # Environment tests (SVAR, ANN)
├── test_agent.py           # DDPG agent tests
├── test_policies.py        # Policy implementation tests
├── test_data.py            # Data loading tests
├── test_metrics.py         # Evaluation metrics tests
├── test_integration.py     # Integration tests
└── test_utils.py           # Utility function tests
```

## Running Tests

Ensure you have your virtual environment activated.

### Run All Tests
```bash
pytest tests/ -v
```

### Run with Coverage
```bash
pytest tests/ -v --cov=src --cov-report=html
```

### Run Specific Test File
```bash
pytest tests/test_environment.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_environment.py::TestSVAREconomy -v
```

### Run Specific Test
```bash
pytest tests/test_environment.py::TestSVAREconomy::test_step_shape -v
```

### Run Fast Tests Only
```bash
pytest tests/ -v -m "not slow"
```

### Run Integration Tests Only
```bash
pytest tests/ -v -m integration
```

## Test Categories

### Unit Tests
- **Environment**: Test SVAR and ANN economies independently
- **Agent**: Test DDPG components (networks, buffer, noise)
- **Policies**: Test baseline and RL policy implementations
- **Data**: Test data loading and preprocessing
- **Metrics**: Test loss and evaluation metrics

### Integration Tests
- **Training Workflow**: Test complete training episodes
- **Evaluation Workflow**: Test policy comparison
- **Save/Load**: Test checkpoint saving and loading

### End-to-End Tests
- **Complete Pipeline**: Test estimate → train → evaluate workflow
- **Multiple Economies**: Test policies across different environments

## Coverage Goals

Target coverage: **>80%** for all modules

Current coverage by module:
- `src/environment/`: ~90%
- `src/agents/`: ~85%
- `src/policies/`: ~95%
- `src/data/`: ~80%
- `src/utils/`: ~75%

## Writing New Tests

### Naming Conventions
- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Example Test
```python
def test_my_feature(fixture_name):
    """Test description."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = my_function(input_data)
    
    # Assert
    assert result == expected_value
```

### Using Fixtures
```python
@pytest.fixture
def my_fixture():
    """Fixture description."""
    return create_test_object()

def test_with_fixture(my_fixture):
    """Test using fixture."""
    assert my_fixture.property == expected_value
```

## Common Issues

### Import Errors
Ensure package is installed in editable mode:
```bash
pip install -e .
```

### Random Seed Issues
Use `set_seeds` fixture for reproducible tests:
```python
def test_reproducible(set_seeds):
    # Test code here
    pass
```

### Slow Tests
Mark slow tests for optional exclusion:
```python
@pytest.mark.slow
def test_slow_operation():
    # Slow test code
    pass
```