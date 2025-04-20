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

  const handleCropImage = () => {
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

    // We need the *natural* dimensions of the rotated image data URL
    // Load it into a temporary image object to get these
    const tempImg = new Image();
    tempImg.onload = () => {
      const naturalWidth = tempImg.naturalWidth;
      const naturalHeight = tempImg.naturalHeight;

       // Determine scale factors between displayed image size and its natural size
      const scaleX = naturalWidth / imageToCrop.width;
      const scaleY = naturalHeight / imageToCrop.height;

      // Calculate crop dimensions in terms of natural image pixels
      let cropX = 0;
      let cropY = 0;
      let cropWidth = 0;
      let cropHeight = 0;

      // Adjust calculation based on crop unit ('%' or 'px')
      if (crop.unit === '%') {
          cropX = (crop.x / 100) * naturalWidth;
          cropY = (crop.y / 100) * naturalHeight;
          cropWidth = (crop.width / 100) * naturalWidth;
          cropHeight = (crop.height / 100) * naturalHeight;
      } else { // Assuming unit is 'px'
          cropX = crop.x * scaleX;
          cropY = crop.y * scaleY;
          cropWidth = crop.width * scaleX;
          cropHeight = crop.height * scaleY;
      }

       // Ensure calculated dimensions are positive
       if (cropWidth <= 0 || cropHeight <= 0) {
            console.error("Calculated zero or negative crop dimensions:", { cropWidth, cropHeight });
            return;
       }

      // Set the final canvas size to the target crop size
      canvas.width = cropWidth;
      canvas.height = cropHeight;

      // Draw the *cropped section* of the rotated image onto the canvas
      // No rotation needed here as the source image is already rotated
      ctx.drawImage(
        tempImg, // Use the temporary image with natural dimensions
        cropX,
        cropY,
        cropWidth,
        cropHeight,
        0, // Destination x on canvas
        0, // Destination y on canvas
        cropWidth, // Destination width on canvas
        cropHeight // Destination height on canvas
      );

      // Get the final cropped image data URL
      try {
        const dataUrl = canvas.toDataURL('image/jpeg');
        if (dataUrl === 'data:,') {
          console.error("Generated empty data URL during final crop.");
        } else {
          setCroppedImage(dataUrl);
          setStage('displaying'); // Move to display stage
        }
      } catch (e) {
        console.error("Error generating final cropped data URL:", e);
      }
    };
    tempImg.onerror = () => {
        console.error("Failed to load rotated image data for cropping.");
    };
    tempImg.src = rotatedImageDataUrl; // Start loading the rotated image data
  };

  const handleEditAgain = () => {
      setCroppedImage(null);
      setRotatedImageDataUrl(null); // Clear rotated image too
      setRotation(0);
      setCrop(createDefaultCrop());
      setStage('rotating'); // Go back to rotating the original image
  }

  return (
    <div className="app-container">
      <h1>Car Sheet OCR</h1>
      <div className="upload-container">
        <label htmlFor="image-upload" className="upload-button">
          {image ? 'Change Image' : 'Upload Image'}
        </label>
        <input
          id="image-upload"
          type="file"
          accept="image/jpeg,image/png"
          onChange={handleImageUpload}
          style={{ display: 'none' }}
        />

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
            onCropImage={handleCropImage}
            rotatedImageRef={rotatedImageRef}
          />
        )}


        {/* Stage: Displaying Result */}
        {stage === 'displaying' && croppedImage && (
          <ImageResult
            croppedImage={croppedImage}
            onEditAgain={handleEditAgain}
          />
        )}

      </div>
    </div>
  )
}

export default App
