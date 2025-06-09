import React, { useState, useEffect, useRef } from 'react';
import * as XLSX from 'xlsx';
import './ImageResult.css';

// Define a type for a single prediction item
interface CodePrediction {
  pred_index: number;
  description: string;
  code_name: string;
  code_coords: [number, number][];
  desc_coords: [number, number][];
}

// Define a type for the API response structure we expect
interface DetectionResponse {
  status: string;
  message: string;
  predictions: any; // We don't use this field directly
  code_descriptions: {
    [code: string]: CodePrediction;
  };
}

interface ImageResultProps {
  croppedImage: string | null;
  onEditAgain: () => void;
  isLoading: boolean;
  detectionResult: DetectionResponse | { error: string } | null;
}

interface PopupState {
  visible: boolean;
  top: number;
  left: number;
}

const ImageResult: React.FC<ImageResultProps> = ({
  croppedImage,
  onEditAgain,
  isLoading,
  detectionResult,
}) => {
  const [editableDescriptions, setEditableDescriptions] = useState<{[code: string]: string}>({});
  const [selectedCode, setSelectedCode] = useState<string | null>(null);
  const [popup, setPopup] = useState<PopupState>({ visible: false, top: 0, left: 0 });
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Effect to parse predictions from props into local state
  useEffect(() => {
    if (detectionResult && !('error' in detectionResult) && detectionResult.code_descriptions) {
      const descriptions: {[code: string]: string} = {};
      Object.entries(detectionResult.code_descriptions).forEach(([code, prediction]) => {
        descriptions[code] = prediction.description;
      });
      setEditableDescriptions(descriptions);
      setSelectedCode(null);
      setPopup({ ...popup, visible: false });
    } else {
      setEditableDescriptions({});
      setPopup({ ...popup, visible: false });
    }
  }, [detectionResult]);

  // Effect to draw highlight on canvas when selection changes or image loads
  useEffect(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    const ctx = canvas?.getContext('2d');

    if (!canvas || !image || !ctx || !detectionResult || 'error' in detectionResult || !detectionResult.code_descriptions) {
      return;
    }

    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Ensure canvas dimensions match displayed image dimensions
    if (canvas.width !== image.clientWidth || canvas.height !== image.clientHeight) {
      canvas.width = image.clientWidth;
      canvas.height = image.clientHeight;
    }

    if (selectedCode && detectionResult.code_descriptions[selectedCode]) {
      const prediction = detectionResult.code_descriptions[selectedCode];
      const coordinates = prediction.desc_coords;

      // Calculate scaling factor
      const scaleX = image.clientWidth / image.naturalWidth;
      const scaleY = image.clientHeight / image.naturalHeight;

      if (coordinates && coordinates.length > 1) {
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.beginPath();
        coordinates.forEach(([x, y], index) => {
          const scaledX = x * scaleX;
          const scaledY = y * scaleY;
          if (index === 0) {
            ctx.moveTo(scaledX, scaledY);
          } else {
            ctx.lineTo(scaledX, scaledY);
          }
        });
        ctx.closePath();
        ctx.stroke();
      }
    }
  }, [selectedCode, detectionResult, imageRef.current?.clientWidth, imageRef.current?.clientHeight]);

  const handleDescriptionChange = (code: string, newDescription: string) => {
    setEditableDescriptions(current => ({
      ...current,
      [code]: newDescription
    }));
  };

  const handleFocus = (code: string, event: React.FocusEvent<HTMLInputElement>) => {
    setSelectedCode(code);
    const inputRect = event.target.getBoundingClientRect();
    setPopup({
      visible: true,
      top: inputRect.top + window.scrollY - 30,
      left: inputRect.left + window.scrollX + inputRect.width / 2,
    });
  };

  const handleBlur = () => {
    setPopup(prev => ({ ...prev, visible: false }));
  };

  const handleImageLoad = () => {
    setSelectedCode(prev => prev);
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (canvas && image) {
      canvas.width = image.clientWidth;
      canvas.height = image.clientHeight;
    }
  };

  const handleDownloadExcel = () => {
    if (!detectionResult || 'error' in detectionResult || !detectionResult.code_descriptions) {
      return;
    }

    // Prepare data for Excel
    const data = Object.entries(detectionResult.code_descriptions).map(([code, prediction]) => ({
      Code: code,
      'Code Name': prediction.code_name,
      'Description': editableDescriptions[code] || prediction.description
    }));

    // Create worksheet
    const ws = XLSX.utils.json_to_sheet(data);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, 'OCR Results');

    // Generate Excel file
    XLSX.writeFile(wb, 'ocr_results.xlsx');
  };

  // Display error message if detection failed
  if (detectionResult && 'error' in detectionResult) {
    return (
      <div className="image-editor result-container error-container">
        <h2>Error</h2>
        <p>No se pudo procesar la imagen:</p>
        <pre>{JSON.stringify(detectionResult.error, null, 2)}</pre>
        <button onClick={onEditAgain} className="action-button">
          Intentar de Nuevo
        </button>
      </div>
    );
  }

  return (
    <div className="image-editor result-container">
      <h2>Resultado</h2>

      <div className="image-display-area" style={{ position: 'relative', maxWidth: '100%' }}>
        {croppedImage && (
          <img
            ref={imageRef}
            src={croppedImage}
            alt="Resultado final recortado"
            style={{ display: 'block', maxWidth: '100%', maxHeight: '500px' }}
            onLoad={handleImageLoad}
          />
        )}
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            pointerEvents: 'none',
          }}
        />
      </div>

      {isLoading && <div className="loading-indicator">Procesando...</div>}

      {!isLoading && detectionResult && !('error' in detectionResult) && detectionResult.code_descriptions && (
        <div className="detection-results-table">
          <div className="table-header">
            <h3>Códigos y Descripciones Detectados</h3>
            <button
              onClick={handleDownloadExcel}
              className="download-button action-button"
            >
              Descargar Excel
            </button>
          </div>
          <table>
            <thead>
              <tr>
                <th>Código</th>
                <th>Nombre del Código</th>
                <th>Descripción</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(detectionResult.code_descriptions).map(([code, prediction]) => (
                <tr key={code}>
                  <td>{code}</td>
                  <td>{prediction.code_name}</td>
                  <td>
                    <input
                      type="text"
                      value={editableDescriptions[code] || ''}
                      onChange={(e) => handleDescriptionChange(code, e.target.value)}
                      onFocus={(e) => handleFocus(code, e)}
                      onBlur={handleBlur}
                      className="description-input"
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!isLoading && (!detectionResult || !('code_descriptions' in detectionResult)) && Object.keys(editableDescriptions).length === 0 && (
        <p>No hay resultados de detección disponibles.</p>
      )}

      {!isLoading && (
        <button
          onClick={onEditAgain}
          className="edit-again-button action-button"
        >
          Comenzar de Nuevo / Editar de Nuevo
        </button>
      )}
    </div>
  );
};

export default ImageResult; 