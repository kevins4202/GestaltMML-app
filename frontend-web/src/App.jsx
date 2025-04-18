// App.js
import React, { useState, useEffect } from "react";
import "./App.css";

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
      setError("Please select an image first");
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await fetch("https://backend-2c1js6jmd-kevins4202s-projects.vercel.app/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Prediction failed");
      }

      const data = await response.json();
      setPredictions(data["predictions"]);
    } catch (err) {
      setError("Error: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    console.log("Predictions:", predictions);
  }, [predictions]);

  return (
    <div className="App">
      <header>
        <h1>Medical Image Diagnosis</h1>
      </header>

      <main>
        <section className="upload-section" aria-label="Upload medical image">
          <div className="upload-container">
            <form
              onSubmit={handleSubmit}
              className="upload-form"
              autoComplete="off"
            >
              <label htmlFor="file-upload" className="file-label">
                Select an image to diagnose:
              </label>
              <input
                id="file-upload"
                className="file-input"
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                aria-label="Upload image file"
              />
              {preview && (
                <div className="preview" aria-label="Image preview">
                  <img src={preview} alt="Preview" />
                </div>
              )}
              <button type="submit" disabled={loading} className="submit-btn">
                {loading ? "Analyzing..." : "Diagnose"}
              </button>
              {error && (
                <div className="error" role="alert">
                  {error}
                </div>
              )}
            </form>
          </div>
        </section>

        {predictions && (
          <section className="results-section" aria-label="Diagnosis results">
            <div className="results">
              <h2>Diagnosis Results</h2>
              <div className="result-content">
                {predictions.map((pred, idx) => (
                  <div className="result-row" key={idx}>
                    <span className="result-label">{pred[0]}:</span>
                    <span className="result-confidence">
                      {(pred[1] * 100).toFixed(2)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
};

export default App;
