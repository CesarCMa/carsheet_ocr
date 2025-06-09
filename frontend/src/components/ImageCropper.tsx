import React from 'react';
import ReactCrop, { Crop } from 'react-image-crop';
import 'react-image-crop/dist/ReactCrop.css';

interface ImageCropperProps {
  rotatedImageDataUrl: string;
  crop: Crop;
  onCropChange: (newCrop: Crop) => void;
  onCropImage: () => void;
  rotatedImageRef: React.RefObject<HTMLImageElement | null>;
}

const ImageCropper: React.FC<ImageCropperProps> = ({
  rotatedImageDataUrl,
  crop,
  onCropChange,
  onCropImage,
  rotatedImageRef,
}) => {
  return (
    <div className="image-editor">
      <h2>Paso 2: Recortar Imagen</h2>
      <div style={{
        display: 'flex', flexDirection: 'row', alignItems: 'center', gap: '1.5rem', marginBottom: '1.5rem', background: '#f8fafc', borderRadius: '8px', padding: '1rem', boxShadow: '0 2px 8px rgba(0,0,0,0.04)'
      }}>
        <img src="/sample_carsheet.png" alt="Ejemplo de recorte de ficha técnica" style={{ maxWidth: '220px', borderRadius: '6px', boxShadow: '0 2px 8px rgba(0,0,0,0.08)' }} />
        <span style={{ fontSize: '1.05rem', color: '#333', fontWeight: 500 }}>
          Recorta las tres columnas principales, incluyendo el número de serie que hay encima de éstas
        </span>
      </div>
      <div className="image-preview">
        <ReactCrop
          crop={crop}
          onChange={onCropChange}
          aspect={undefined} // Or set a specific aspect ratio if desired
        >
          {/* Display the already rotated image */}
          <img
            ref={rotatedImageRef} // Ref for the rotated image being cropped
            src={rotatedImageDataUrl}
            alt="Imagen rotada para recortar"
            // No inline rotation style needed here
          />
        </ReactCrop>
      </div>
      <div className="image-controls">
        <button onClick={onCropImage} className="crop-button">
          Recortar Imagen
        </button>
      </div>
    </div>
  );
};

export default ImageCropper; 