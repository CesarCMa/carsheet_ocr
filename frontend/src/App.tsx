import { useState } from 'react'
import './App.css'

function App() {
  const [image, setImage] = useState<string | null>(null)

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => {
        setImage(reader.result as string)
      }
      reader.readAsDataURL(file)
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
          <div className="image-preview">
            <img src={image} alt="Uploaded preview" />
          </div>
        )}
      </div>
    </div>
  )
}

export default App
