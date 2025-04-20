import { useState, useRef } from 'react'
import ReactCrop, { Crop } from 'react-image-crop'
import 'react-image-crop/dist/ReactCrop.css'
import './App.css'

function App() {
  const [image, setImage] = useState<string | null>(null)
  const [crop, setCrop] = useState<Crop>({
    unit: '%',
    width: 90,
    height: 90,
    x: 5,
    y: 5
  })
  const [rotation, setRotation] = useState(0)
  const [croppedImage, setCroppedImage] = useState<string | null>(null)
  const imageRef = useRef<HTMLImageElement>(null)

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => {
        setImage(reader.result as string)
        setCroppedImage(null)
        setRotation(0)
        setCrop({
          unit: '%',
          width: 90,
          height: 90,
          x: 5,
          y: 5
        })
      }
      reader.readAsDataURL(file)
    }
  }

  const handleRotate = (angle: number) => {
    setRotation(angle)
  }

  const handleCropComplete = (crop: Crop) => {
    setCrop(crop)
  }

  const handleCropImage = () => {
    if (imageRef.current && crop.width && crop.height) {
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      if (!ctx) return

      const image = imageRef.current
      const scaleX = image.naturalWidth / image.width
      const scaleY = image.naturalHeight / image.height

      // Calculate the actual pixel values for the crop
      const cropX = (crop.x / 100) * image.naturalWidth
      const cropY = (crop.y / 100) * image.naturalHeight
      const cropWidth = (crop.width / 100) * image.naturalWidth
      const cropHeight = (crop.height / 100) * image.naturalHeight

      // Set canvas size to match the crop dimensions
      canvas.width = cropWidth
      canvas.height = cropHeight

      // Apply rotation
      ctx.translate(canvas.width / 2, canvas.height / 2)
      ctx.rotate((rotation * Math.PI) / 180)
      ctx.translate(-canvas.width / 2, -canvas.height / 2)

      // Draw the cropped portion of the image
      ctx.drawImage(
        image,
        cropX,
        cropY,
        cropWidth,
        cropHeight,
        0,
        0,
        cropWidth,
        cropHeight
      )

      setCroppedImage(canvas.toDataURL('image/jpeg'))
    }
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
        {image && (
          <div className="image-editor">
            {!croppedImage ? (
              <>
                <div className="image-preview">
                  <ReactCrop
                    crop={crop}
                    onChange={(c) => setCrop(c)}
                    onComplete={handleCropComplete}
                    aspect={undefined}
                  >
                    <img
                      ref={imageRef}
                      src={image}
                      alt="Uploaded preview"
                      style={{ transform: `rotate(${rotation}deg)` }}
                    />
                  </ReactCrop>
                </div>
                <div className="image-controls">
                  <div className="rotation-control">
                    <label htmlFor="rotation-slider">Rotation: {rotation}°</label>
                    <input
                      id="rotation-slider"
                      type="range"
                      min="0"
                      max="360"
                      value={rotation}
                      onChange={(e) => handleRotate(Number(e.target.value))}
                      className="rotation-slider"
                    />
                  </div>
                  <button onClick={handleCropImage} className="crop-button">
                    Crop Image
                  </button>
                </div>
              </>
            ) : (
              <div className="cropped-result">
                <img src={croppedImage} alt="Cropped result" />
                <button 
                  onClick={() => setCroppedImage(null)} 
                  className="edit-again-button"
                >
                  Edit Again
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
