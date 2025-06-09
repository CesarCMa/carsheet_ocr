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