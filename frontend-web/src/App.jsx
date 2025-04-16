// App.js
import React, { useState, useEffect } from 'react';
import './App.css';

/**
 * App component for medical image diagnosis.
 * - Manages state for selected file, preview image, prediction results, loading, and error messages.
 * - Handles file selection and creates a preview using FileReader.
 * - Submits the selected image to a backend API for prediction.
 * - Displays error messages if the file is not selected or if prediction fails.
 * - Shows loading state while prediction is in progress.
 * - Renders the diagnosis results with prediction and confidence levels.
 */
const App = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
      
      // Reset results
      setPredictions(null);
      setError(null);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await fetch('http://localhost:5001/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const data = await response.json();
      setPredictions(data['predictions']);
    } catch (err) {
      setError('Error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    console.log('Predictions:', predictions);
  }, [predictions]);

  return (
    <div className="App">
      <header>
        <h1>Medical Image Diagnosis</h1>
      </header>
      
      <main>
        <div className="upload-container">
          <h2>Upload Image</h2>
          <form onSubmit={handleSubmit}>
            <div className="file-input">
              <input 
                type="file" 
                onChange={handleFileChange} 
                accept="image/*"
              />
            </div>
            
            {preview && (
              <div className="preview">
                <h3>Preview</h3>
                <img src={preview} alt="Preview" />
              </div>
            )}
            
            <button 
              type="submit" 
              disabled={!selectedFile || loading}
            >
              {loading ? 'Processing...' : 'Get Diagnosis'}
            </button>
          </form>
          
          {error && <div className="error">{error}</div>}
        </div>
        
        {predictions && (
          <div className="results">
            <h2>Diagnosis Results</h2>
            <div className="result-content">
              {predictions.map((prediction, index) => (
                <div key={index}>
                  <p><strong>Disease:</strong> {prediction[0]}</p>
                  <p><strong>Confidence:</strong> {prediction[1]}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;