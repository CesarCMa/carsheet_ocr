import React from 'react';

interface ImageResultProps {
  croppedImage: string;
  onEditAgain: () => void;
}

const ImageResult: React.FC<ImageResultProps> = ({
  croppedImage,
  onEditAgain,
}) => {
  return (
    <div className="image-editor">
      <h2>Result</h2>
      <div className="cropped-result">
        <img src={croppedImage} alt="Final cropped result" />
        <button
          onClick={onEditAgain}
          className="edit-again-button"
        >
          Edit Again
        </button>
      </div>
    </div>
  );
};

export default ImageResult; 