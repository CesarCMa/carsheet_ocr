import React from 'react';

interface ImageRotatorProps {
  image: string;
  rotation: number;
  onRotate: (angle: number) => void;
  onAcceptRotation: () => void;
  imageRef: React.RefObject<HTMLImageElement | null>;
}

const ImageRotator: React.FC<ImageRotatorProps> = ({
  image,
  rotation,
  onRotate,
  onAcceptRotation,
  imageRef,
}) => {
  return (
    <div className="image-editor">
      <h2>Paso 1: Rotar Imagen</h2>
      <div className="image-preview">
        <div className="image-preview-container"> {/* Container for overlay */}
          <img
            ref={imageRef}
            src={image}
            alt="Vista previa para rotación"
            style={{ transform: `rotate(${rotation}deg)` }}
          />
          <div className="cross-overlay"></div> {/* The cross overlay */}
        </div>
      </div>
      <div className="image-controls">
        <div className="rotation-control">
          <label htmlFor="rotation-slider">Rotación: {rotation}°</label>
          <input
            id="rotation-slider"
            type="range" min="0" max="360" value={rotation}
            onChange={(e) => onRotate(Number(e.target.value))}
            className="rotation-slider"
          />
        </div>
        <button onClick={onAcceptRotation} className="crop-button">
          Aceptar Rotación y Proceder a Recortar
        </button>
      </div>
    </div>
  );
};

export default ImageRotator; 