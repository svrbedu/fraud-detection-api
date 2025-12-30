import pytest
from unittest.mock import patch, MagicMock
import train_model

def test_train_creates_model_when_not_exists():
    """
    Verifies that if the model file does not exist:
    1. A new XGBClassifier is initialized.
    2. The model is fitted.
    3. The model is saved to disk.
    """
    with patch('train_model.os.path.exists') as mock_exists, \
         patch('train_model.joblib.dump') as mock_dump, \
         patch('train_model.xgb.XGBClassifier') as mock_xgb:
        
        # Setup: Simulate that the file does NOT exist
        mock_exists.return_value = False
        
        # Mock the model instance returned by XGBClassifier()
        mock_model_instance = MagicMock()
        mock_xgb.return_value = mock_model_instance
        
        # Execute
        train_model.train()
        
        # Verify
        mock_exists.assert_called_with("fraud_model.joblib")
        mock_xgb.assert_called_once()
        mock_model_instance.fit.assert_called_once()
        mock_dump.assert_called_once_with(mock_model_instance, "fraud_model.joblib")

def test_train_skips_when_model_exists(capsys):
    """
    Verifies that if the model file already exists:
    1. Training is skipped.
    2. No file is written.
    3. A message is printed to stdout.
    """
    with patch('train_model.os.path.exists') as mock_exists, \
         patch('train_model.joblib.dump') as mock_dump, \
         patch('train_model.xgb.XGBClassifier') as mock_xgb:
        
        # Setup: Simulate that the file DOES exist
        mock_exists.return_value = True
        
        # Execute
        train_model.train()
        
        # Verify
        mock_exists.assert_called_with("fraud_model.joblib")
        mock_xgb.assert_not_called()
        mock_dump.assert_not_called()
        
        # Verify output message
        captured = capsys.readouterr()
        assert "Model already exists. Skipping training." in captured.out