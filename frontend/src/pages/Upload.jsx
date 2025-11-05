/**
 * Upload page component
 */

import { useState, useRef } from 'react'
import { HiCloudUpload, HiDocument, HiCheckCircle, HiExclamation, HiX } from 'react-icons/hi'
import { uploadPDF } from '../services/api'
import './Upload.css'

function Upload() {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadSuccess, setUploadSuccess] = useState(null)
  const [uploadError, setUploadError] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef(null)

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = (selectedFile) => {
    // Validate file type
    if (!selectedFile.name.toLowerCase().endsWith('.pdf')) {
      setUploadError('Please select a PDF file')
      return
    }

    // Validate file size (50MB limit)
    const maxSize = 50 * 1024 * 1024 // 50MB
    if (selectedFile.size > maxSize) {
      setUploadError('File size must be less than 50MB')
      return
    }

    setFile(selectedFile)
    setUploadError(null)
    setUploadSuccess(null)
  }

  const handleUpload = async () => {
    if (!file) return

    setUploading(true)
    setUploadError(null)

    try {
      const result = await uploadPDF(file)
      setUploadSuccess(result)
      setFile(null)
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    } catch (error) {
      setUploadError(error.response?.data?.detail || error.message || 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  const handleRemoveFile = () => {
    setFile(null)
    setUploadError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i]
  }

  return (
    <div className="upload-page">
      <div className="upload-container">
        {/* Upload Zone */}
        <div
          className={`upload-zone ${dragActive ? 'drag-active' : ''} ${file ? 'has-file' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => !file && fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            onChange={handleFileInput}
            style={{ display: 'none' }}
          />

          {!file ? (
            <>
              <div className="upload-icon">
                <HiCloudUpload />
              </div>
              <div className="upload-title">Drop PDF file here or click to browse</div>
              <div className="upload-subtitle">Supports LAQ PDFs up to 50MB</div>
            </>
          ) : (
            <div className="file-preview">
              <div className="file-icon">
                <HiDocument />
              </div>
              <div className="file-info">
                <div className="file-name">{file.name}</div>
                <div className="file-size">{formatFileSize(file.size)}</div>
              </div>
              <button
                className="file-remove"
                onClick={(e) => {
                  e.stopPropagation()
                  handleRemoveFile()
                }}
                disabled={uploading}
              >
                <HiX />
              </button>
            </div>
          )}
        </div>

        {/* Upload Button */}
        {file && (
          <button
            className="upload-button"
            onClick={handleUpload}
            disabled={uploading}
          >
            <HiCloudUpload />
            <span>{uploading ? 'Processing...' : 'Upload and Process PDF'}</span>
          </button>
        )}

        {/* Error Message */}
        {uploadError && (
          <div className="message error-message">
            <HiExclamation />
            <span>{uploadError}</span>
          </div>
        )}

        {/* Success Message */}
        {uploadSuccess && (
          <div className="message success-message">
            <HiCheckCircle />
            <div className="success-content">
              <div className="success-title">Successfully processed {uploadSuccess.pdf_name}</div>
              <div className="success-details">
                Extracted {uploadSuccess.qa_pairs_extracted} Q&A pairs
              </div>
            </div>
          </div>
        )}

        {/* Extracted Data Preview */}
        {uploadSuccess && uploadSuccess.laq_data && (
          <div className="extracted-data">
            <div className="section-header">
              <h3 className="section-title">Extracted LAQ Data</h3>
            </div>

            <div className="laq-metadata">
              <div className="metadata-item">
                <span className="metadata-label">Title:</span>
                <span className="metadata-value">{uploadSuccess.laq_data.pdf_title}</span>
              </div>
              <div className="metadata-item">
                <span className="metadata-label">LAQ Number:</span>
                <span className="metadata-value">{uploadSuccess.laq_data.laq_number}</span>
              </div>
              <div className="metadata-item">
                <span className="metadata-label">Type:</span>
                <span className="metadata-value">{uploadSuccess.laq_data.laq_type}</span>
              </div>
              <div className="metadata-item">
                <span className="metadata-label">Minister:</span>
                <span className="metadata-value">{uploadSuccess.laq_data.minister}</span>
              </div>
              <div className="metadata-item">
                <span className="metadata-label">Date:</span>
                <span className="metadata-value">{uploadSuccess.laq_data.date}</span>
              </div>
            </div>

            <div className="qa-pairs">
              <h4 className="qa-title">Q&A Pairs ({uploadSuccess.laq_data.qa_pairs.length})</h4>
              {uploadSuccess.laq_data.qa_pairs.slice(0, 3).map((qa, index) => (
                <div key={index} className="qa-pair">
                  <div className="question">
                    <strong>Q{index + 1}:</strong> {qa.question}
                  </div>
                  <div className="answer">
                    <strong>A:</strong> {qa.answer}
                  </div>
                </div>
              ))}
              {uploadSuccess.laq_data.qa_pairs.length > 3 && (
                <div className="qa-more">
                  +{uploadSuccess.laq_data.qa_pairs.length - 3} more Q&A pairs
                </div>
              )}
            </div>
          </div>
        )}

        {/* Info Section */}
        <div className="upload-info">
          <h3 className="info-title">What happens when you upload?</h3>
          <div className="info-steps">
            <div className="info-step">
              <div className="step-number">1</div>
              <div className="step-content">
                <div className="step-title">PDF Processing</div>
                <div className="step-description">Convert PDF to structured markdown</div>
              </div>
            </div>
            <div className="info-step">
              <div className="step-number">2</div>
              <div className="step-content">
                <div className="step-title">LLM Extraction</div>
                <div className="step-description">Extract Q&A pairs using Mistral</div>
              </div>
            </div>
            <div className="info-step">
              <div className="step-number">3</div>
              <div className="step-content">
                <div className="step-title">Generate Embeddings</div>
                <div className="step-description">Create vector embeddings for semantic search</div>
              </div>
            </div>
            <div className="info-step">
              <div className="step-number">4</div>
              <div className="step-content">
                <div className="step-title">Store in Database</div>
                <div className="step-description">Save to ChromaDB for fast retrieval</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Upload
