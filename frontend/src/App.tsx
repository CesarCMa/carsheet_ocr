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
  const imageRef = useRef<HTMLImageElement>(null)

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => {
        setImage(reader.result as string)
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
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
