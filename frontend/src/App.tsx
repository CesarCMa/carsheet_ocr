import { useState, useRef } from 'react'
import { Crop, centerCrop, makeAspectCrop } from 'react-image-crop'
import 'react-image-crop/dist/ReactCrop.css'
import './App.css'
import ImageRotator from './components/ImageRotator';
import ImageCropper from './components/ImageCropper';
import ImageResult from './components/ImageResult';

// Define the stages of the image editing workflow
type Stage = 'uploading' | 'rotating' | 'cropping' | 'displaying';

// Helper function to create a default centered crop
function createDefaultCrop(aspect?: number): Crop {
  return centerCrop(
    makeAspectCrop(
      {
        unit: '%',
        width: 90,
      },
      aspect || 1, // Default to aspect 1 if none provided
      100, // Assuming image width is 100% for initial calc
      100  // Assuming image height is 100% for initial calc
    ),
    100,
    100
  );
}

function App() {
  const [image, setImage] = useState<string | null>(null)
  const [crop, setCrop] = useState<Crop>(createDefaultCrop())
  const [rotation, setRotation] = useState(0)
  const [croppedImage, setCroppedImage] = useState<string | null>(null)
  const [rotatedImageDataUrl, setRotatedImageDataUrl] = useState<string | null>(null)
  const [stage, setStage] = useState<Stage>('uploading')
  const [detectionResult, setDetectionResult] = useState<any | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const imageRef = useRef<HTMLImageElement>(null)
  const rotatedImageRef = useRef<HTMLImageElement>(null)

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => {
        const result = reader.result as string;
        setImage(result)
        setCroppedImage(null)
        setRotatedImageDataUrl(null)
        setRotation(0)
        setCrop(createDefaultCrop())
        setStage('rotating')
      }
      reader.readAsDataURL(file)
    }
  }

  const handleRotate = (angle: number) => {
    setRotation(angle)
  }

  const handleCropChange = (newCrop: Crop) => {
    setCrop(newCrop);
  };

  const handleAcceptRotation = () => {
    const sourceImage = imageRef.current;
    if (!sourceImage || !image) return;

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rad = (rotation * Math.PI) / 180;

    // Calculate canvas size needed to contain the rotated image
    const newWidth = Math.abs(sourceImage.naturalWidth * Math.cos(rad)) + Math.abs(sourceImage.naturalHeight * Math.sin(rad));
    const newHeight = Math.abs(sourceImage.naturalWidth * Math.sin(rad)) + Math.abs(sourceImage.naturalHeight * Math.cos(rad));

    canvas.width = newWidth;
    canvas.height = newHeight;

    // Move context to center, rotate, and move back
    ctx.translate(newWidth / 2, newHeight / 2);
    ctx.rotate(rad);
    // Draw the image centered in the rotated context
    ctx.drawImage(sourceImage, -sourceImage.naturalWidth / 2, -sourceImage.naturalHeight / 2, sourceImage.naturalWidth, sourceImage.naturalHeight);

    // Save the rotated image data URL
    const rotatedUrl = canvas.toDataURL('image/png'); // Use PNG to preserve transparency if needed
    setRotatedImageDataUrl(rotatedUrl);
    setRotation(0); // Reset rotation as it's baked into the image now
    setCrop(createDefaultCrop()) // Reset crop for the new image
    setStage('cropping'); // Move to cropping stage
  };

  // Helper function to convert data URL to Blob
  const dataURLtoBlob = (dataurl: string): Blob | null => {
      const arr = dataurl.split(',');
      const match = arr[0].match(/:(.*?);/);
      if (!match) return null;
      const mime = match[1];
      const bstr = atob(arr[1]);
      let n = bstr.length;
      const u8arr = new Uint8Array(n);
      while(n--){
          u8arr[n] = bstr.charCodeAt(n);
      }
      return new Blob([u8arr], {type:mime});
  }

  // Function to call the backend API
  const callDetectAPI = async (imageDataUrl: string) => {
    const blob = dataURLtoBlob(imageDataUrl);
    if (!blob) {
        console.error("Failed to convert data URL to Blob.");
        setDetectionResult({ error: "Failed to process image before sending." });
        return;
    }

    const formData = new FormData();
    // Use a generic filename, the backend should handle it based on content type
    formData.append('file', blob, 'image.jpg');

    setIsLoading(true);
    setDetectionResult(null); // Clear previous results

    try {
        const response = await fetch('http://127.0.0.1:8000/detect', {
            method: 'POST',
            body: formData,
            // Headers might not be needed if the server correctly handles multipart/form-data
            // headers: { 'Content-Type': 'multipart/form-data' } // Fetch usually sets this automatically for FormData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        setDetectionResult(result);

    } catch (error) {
        console.error("Error calling detect API:", error);
        setDetectionResult({ error: `Failed to fetch detection results: ${error}` });
    } finally {
        setIsLoading(false);
    }
  };

  // --- Modified handleCropImage that calls the API ---
  const handleCropImageAndDetect = () => {
    const imageToCrop = rotatedImageRef.current;

    if (!imageToCrop || !rotatedImageDataUrl || !crop.width || !crop.height) {
      console.error("Rotated image or crop dimensions missing for cropping.");
      return;
    }

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.error("Failed to get canvas context for cropping.");
      return;
    }

    const tempImg = new Image();
    tempImg.onload = () => {
      const naturalWidth = tempImg.naturalWidth;
      const naturalHeight = tempImg.naturalHeight;
      const scaleX = naturalWidth / imageToCrop.width;
      const scaleY = naturalHeight / imageToCrop.height;

      let cropX = 0, cropY = 0, cropWidth = 0, cropHeight = 0;
      if (crop.unit === '%') {
          cropX = (crop.x / 100) * naturalWidth;
          cropY = (crop.y / 100) * naturalHeight;
          cropWidth = (crop.width / 100) * naturalWidth;
          cropHeight = (crop.height / 100) * naturalHeight;
      } else {
          cropX = crop.x * scaleX;
          cropY = crop.y * scaleY;
          cropWidth = crop.width * scaleX;
          cropHeight = crop.height * scaleY;
      }

      if (cropWidth <= 0 || cropHeight <= 0) {
           console.error("Calculated zero or negative crop dimensions:", { cropWidth, cropHeight });
           return;
      }

      canvas.width = cropWidth;
      canvas.height = cropHeight;

      ctx.drawImage(tempImg, cropX, cropY, cropWidth, cropHeight, 0, 0, cropWidth, cropHeight);

      try {
        const dataUrl = canvas.toDataURL('image/jpeg'); // Use JPEG for potentially smaller size
        if (dataUrl === 'data:,') {
          console.error("Generated empty data URL during final crop.");
        } else {
          setCroppedImage(dataUrl);
          setStage('displaying'); // Move to display stage
          callDetectAPI(dataUrl); // <<< Call the API here
        }
      } catch (e) {
        console.error("Error generating final cropped data URL:", e);
      }
    };
    tempImg.onerror = () => {
        console.error("Failed to load rotated image data for cropping.");
    };
    tempImg.src = rotatedImageDataUrl;
  };

  const handleEditAgain = () => {
      setCroppedImage(null);
      setRotatedImageDataUrl(null); // Clear rotated image too
      setDetectionResult(null); // Clear detection results
      setIsLoading(false); // Reset loading state
      setRotation(0);
      setCrop(createDefaultCrop());
      setStage('rotating'); // Go back to rotating the original image
  }

  return (
    <div className="app-container">
      <nav className="navbar">
        <div className="navbar-brand">FichaScan</div>
        <div className="navbar-credit">
          Desarrollado por <a href="https://github.com/CesarCMa" target="_blank" rel="noopener noreferrer">CesarCMa</a>
        </div>
      </nav>

      {stage === 'uploading' ? (
        <div className="main-grid">
          <div className="main-left">
            <h2 className="main-header">¡Convierte tus fichas técnicas a excel!</h2>
            <p className="main-subtitle">FichaScan te permite escanear fichas técnicas de vehículos y convertirlas a excel 🪄</p>
            <div className="main-upload-btn-container">
              <label htmlFor="image-upload" className="upload-button">
                Escanear Ficha
              </label>
              <input
                id="image-upload"
                type="file"
                accept="image/jpeg,image/png"
                onChange={handleImageUpload}
                style={{ display: 'none' }}
              />
            </div>
            <div className="main-warning">
              Actualmente solo soportamos fichas técnicas electrónicas (posteriores a 2016)
            </div>
          </div>
          <div className="main-right">
            <img src="/main_page_pic.png" alt="Ejemplo de conversión de ficha técnica a excel" className="main-page-image" />
          </div>
        </div>
      ) : (
        <div className="upload-container">
          {/* Stage: Rotating */}
          {stage === 'rotating' && image && (
            <ImageRotator
              image={image}
              rotation={rotation}
              onRotate={handleRotate}
              onAcceptRotation={handleAcceptRotation}
              imageRef={imageRef}
            />
          )}

          {/* Stage: Cropping */}
          {stage === 'cropping' && rotatedImageDataUrl && (
            <ImageCropper
              rotatedImageDataUrl={rotatedImageDataUrl}
              crop={crop}
              onCropChange={handleCropChange}
              onCropImage={handleCropImageAndDetect}
              rotatedImageRef={rotatedImageRef}
            />
          )}

          {/* Stage: Displaying Result */}
          {stage === 'displaying' && (croppedImage || isLoading) && (
            <ImageResult
              croppedImage={croppedImage}
              onEditAgain={handleEditAgain}
              isLoading={isLoading}
              detectionResult={detectionResult}
            />
          )}
        </div>
      )}
    </div>
  )
}

export default App
