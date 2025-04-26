import React, { useState, useEffect, useRef } from 'react';

// Define a type for a single prediction item
interface Prediction {
  bbox: [number, number][]; // [[x1, y1], [x2, y2], ...]
  text: string;
  confidence: number;
}

// Define a type for the API response structure we expect
interface DetectionResponse {
  status?: string;
  message?: string;
  predictions?: [ [number, number][], [string, number] ][];
}

interface ImageResultProps {
  croppedImage: string | null;
  onEditAgain: () => void;
  isLoading: boolean;
  detectionResult: DetectionResponse | { error: string } | null; // Use the defined type
}

const ImageResult: React.FC<ImageResultProps> = ({
  croppedImage,
  onEditAgain,
  isLoading,
  detectionResult,
}) => {
  const [editablePredictions, setEditablePredictions] = useState<Prediction[]>([]);
  const [selectedPredictionIndex, setSelectedPredictionIndex] = useState<number | null>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Effect to parse predictions from props into local state
  useEffect(() => {
    if (detectionResult && 'predictions' in detectionResult && detectionResult.predictions) {
      const parsedPredictions: Prediction[] = detectionResult.predictions.map((p) => ({
        bbox: p[0],
        text: p[1][0],
        confidence: p[1][1],
      }));
      setEditablePredictions(parsedPredictions);
      setSelectedPredictionIndex(null); // Reset selection on new results
    } else {
      setEditablePredictions([]); // Clear predictions if result is invalid or an error
    }
  }, [detectionResult]);

  // Effect to draw highlight on canvas when selection changes or image loads
  useEffect(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    const ctx = canvas?.getContext('2d');

    if (!canvas || !image || !ctx) {
      // console.log("Canvas, image, or context not ready");
      return;
    }

    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Ensure canvas dimensions match displayed image dimensions
    // This needs to happen *after* the image has loaded and rendered
    if (canvas.width !== image.clientWidth || canvas.height !== image.clientHeight) {
        canvas.width = image.clientWidth;
        canvas.height = image.clientHeight;
        // console.log(`Resized canvas to ${canvas.width}x${canvas.height}`);
    }

    if (selectedPredictionIndex !== null && editablePredictions[selectedPredictionIndex]) {
      const prediction = editablePredictions[selectedPredictionIndex];
      const coordinates = prediction.bbox;

      // Calculate scaling factor: prediction coords are relative to natural image size
      const scaleX = image.clientWidth / image.naturalWidth;
      const scaleY = image.clientHeight / image.naturalHeight;

      // console.log(`Image natural: ${image.naturalWidth}x${image.naturalHeight}, client: ${image.clientWidth}x${image.clientHeight}, scale: ${scaleX}x${scaleY}`);

      if (coordinates && coordinates.length > 1) {
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.beginPath();
        // console.log("Drawing polygon:");
        coordinates.forEach(([x, y], index) => {
          const scaledX = x * scaleX;
          const scaledY = y * scaleY;
          // console.log(`  Point ${index}: (${x}, ${y}) -> (${scaledX}, ${scaledY})`);
          if (index === 0) {
            ctx.moveTo(scaledX, scaledY);
          } else {
            ctx.lineTo(scaledX, scaledY);
          }
        });
        ctx.closePath();
        ctx.stroke();
        // console.log("Polygon drawn.");
      } else {
        // console.log("Invalid coordinates for drawing");
      }
    }
    // Dependency array includes selected index and image client dimensions (implicitly via image ref)
    // Adding editablePredictions ensures redraw if coordinates were somehow changed (though not implemented yet)
  }, [selectedPredictionIndex, editablePredictions, imageRef.current?.clientWidth, imageRef.current?.clientHeight]);

  const handlePredictionTextChange = (index: number, newText: string) => {
    setEditablePredictions(currentPredictions =>
      currentPredictions.map((p, i) =>
        i === index ? { ...p, text: newText } : p
      )
    );
  };

  // Handle focus on input to select prediction
  const handleFocus = (index: number) => {
    setSelectedPredictionIndex(index);
  };

  // Handle image load event to trigger initial canvas sizing and drawing
  const handleImageLoad = () => {
      // console.log("Image loaded, triggering effect redraw");
      // Force redraw by updating a state that the effect depends on
      // A simple way is to re-set the selected index
      setSelectedPredictionIndex(prev => prev);
      // Alternatively, trigger a resize event or manage a separate load state
       const canvas = canvasRef.current;
       const image = imageRef.current;
       if (canvas && image) {
            canvas.width = image.clientWidth;
            canvas.height = image.clientHeight;
            // console.log(`Canvas sized on load: ${canvas.width}x${canvas.height}`);
            // Manually trigger draw after sizing
             const ctx = canvas.getContext('2d');
             if (ctx && selectedPredictionIndex !== null && editablePredictions[selectedPredictionIndex]) {
                 // Redraw logic here (duplicate or refactor draw logic)
             }
       }
  };

  // Display error message if detection failed
  if (detectionResult && 'error' in detectionResult) {
    return (
        <div className="image-editor result-container error-container">
            <h2>Error</h2>
            <p>Could not process image:</p>
            <pre>{JSON.stringify(detectionResult.error, null, 2)}</pre>
            <button onClick={onEditAgain} className="action-button">
                Try Again
            </button>
        </div>
    );
  }

  return (
    <div className="image-editor result-container">
      <h2>Result</h2>

      <div className="image-display-area" style={{ position: 'relative', maxWidth: '100%' }}>
        {/* Display Cropped Image */} 
        {croppedImage && (
          <img
            ref={imageRef}
            src={croppedImage}
            alt="Final cropped result"
            style={{ display: 'block', maxWidth: '100%', maxHeight: '500px' }} // Max height added
            onLoad={handleImageLoad} // Add onLoad handler
           />
        )}
        {/* Canvas for highlighting */} 
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            pointerEvents: 'none', // Make canvas ignore mouse events
          }}
        />
      </div>

      {/* Loading Indicator */} 
      {isLoading && <div className="loading-indicator">Processing...</div>}

      {/* Detection Result Display & Editing */} 
      {!isLoading && editablePredictions.length > 0 && (
        <div className="detection-results-list">
          <h3>Detected Text (Editable)</h3>
          <ul>
            {editablePredictions.map((prediction, index) => (
              <li key={index} className={index === selectedPredictionIndex ? 'selected' : ''}>
                <input
                  type="text"
                  value={prediction.text}
                  onChange={(e) => handlePredictionTextChange(index, e.target.value)}
                  onFocus={() => handleFocus(index)}
                  // onClick={() => handleFocus(index)} // Optional: highlight on click too
                  className="prediction-input"
                />
                <span className="confidence-score">
                  ({(prediction.confidence * 100).toFixed(1)}%)
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* No Results Message */} 
      {!isLoading && !detectionResult && editablePredictions.length === 0 && (
         <p>No detection results available.</p>
      )}
      {!isLoading && detectionResult && 'predictions' in detectionResult && editablePredictions.length === 0 && (
          <p>No text detected in the cropped area.</p>
      )
      }

      {/* Edit Again Button */} 
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