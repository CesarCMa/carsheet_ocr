import React from 'react';

interface ImageResultProps {
  croppedImage: string | null;
  onEditAgain: () => void;
  isLoading: boolean;
  detectionResult: any | null;
}

const ImageResult: React.FC<ImageResultProps> = ({
  croppedImage,
  onEditAgain,
  isLoading,
  detectionResult,
}) => {
  return (
    <div className="image-editor result-container">
      <h2>Result</h2>

      {croppedImage && (
        <div className="cropped-image-container">
          <img src={croppedImage} alt="Final cropped result" style={{ maxWidth: '100%', maxHeight: '400px' }}/>
        </div>
      )}

      {isLoading && <div className="loading-indicator">Processing...</div>}

      {detectionResult && (
        <div className="detection-result">
          <h3>Detection Output:</h3>
          <pre><code>{JSON.stringify(detectionResult, null, 2)}</code></pre>
        </div>
      )}

      {!isLoading && (
          <button
            onClick={onEditAgain}
            className="edit-again-button action-button"
          >
            Start Over / Edit Again
          </button>
      )}
    </div>
  );
};

export default ImageResult; 