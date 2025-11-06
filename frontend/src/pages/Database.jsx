/**
 * Database page component
 */

import { useState, useEffect } from 'react'
import { HiDatabase, HiChip, HiDocumentText, HiCollection, HiClock, HiCog } from 'react-icons/hi'
import { HiServerStack } from 'react-icons/hi2'
import { SiOpenai } from 'react-icons/si'
import { getDatabaseInfo } from '../services/api'
import './Database.css'

function Database() {
  const [dbInfo, setDbInfo] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchDbInfo = async () => {
      try {
        const data = await getDatabaseInfo()
        setDbInfo(data)
      } catch (err) {
        setError(err.message || 'Failed to load database information')
        console.error('Database info error:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchDbInfo()
  }, [])

  if (loading) {
    return (
      <div className="database-page">
        <div className="loading-message">Loading database information...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="database-page">
        <div className="error-container">
          <div className="error-icon">
            <HiDatabase />
          </div>
          <div className="error-title">Failed to Load Database Info</div>
          <div className="error-message">{error}</div>
          <button className="retry-button" onClick={() => window.location.reload()}>
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="database-page">
      <div className="database-container">
        {/* Overview Section */}
        <section className="db-section">
          <div className="section-header">
            <h2 className="section-title">Database Overview</h2>
          </div>

          <div className="db-overview-grid">
            <div className="db-card">
              <div className="db-card-icon">
                <HiServerStack />
              </div>
              <div className="db-card-content">
                <div className="db-card-label">Vector Database</div>
                <div className="db-card-value">ChromaDB</div>
                <div className="db-card-description">Cosine similarity search</div>
              </div>
            </div>

            <div className="db-card">
              <div className="db-card-icon">
                <HiCollection />
              </div>
              <div className="db-card-content">
                <div className="db-card-label">Collection Name</div>
                <div className="db-card-value">{dbInfo?.collection_name}</div>
                <div className="db-card-description">LAQ storage collection</div>
              </div>
            </div>

            <div className="db-card">
              <div className="db-card-icon">
                <HiDocumentText />
              </div>
              <div className="db-card-content">
                <div className="db-card-label">Total Documents</div>
                <div className="db-card-value">{dbInfo?.total_documents?.toLocaleString() || 0}</div>
                <div className="db-card-description">Q&A pairs stored</div>
              </div>
            </div>

            <div className="db-card">
              <div className="db-card-icon">
                <HiDatabase />
              </div>
              <div className="db-card-content">
                <div className="db-card-label">Database Path</div>
                <div className="db-card-value path-value">{dbInfo?.database_path}</div>
                <div className="db-card-description">Local storage location</div>
              </div>
            </div>
          </div>
        </section>

        {/* Models Section */}
        <section className="db-section">
          <div className="section-header">
            <h2 className="section-title">AI Models Configuration</h2>
          </div>

          <div className="models-grid">
            <div className="model-card">
              <div className="model-icon">
                <SiOpenai />
              </div>
              <div className="model-content">
                <div className="model-title">LLM Model</div>
                <div className="model-name">{dbInfo?.llm_model}</div>
                <div className="model-description">
                  Language model for answer generation and Q&A extraction
                </div>
                <div className="model-tags">
                  <span className="model-tag">Generation</span>
                  <span className="model-tag">Extraction</span>
                </div>
              </div>
            </div>

            <div className="model-card">
              <div className="model-icon">
                <HiChip />
              </div>
              <div className="model-content">
                <div className="model-title">Embedding Model</div>
                <div className="model-name">{dbInfo?.embedding_model}</div>
                <div className="model-description">
                  Vector embedding model for semantic search capabilities
                </div>
                <div className="model-tags">
                  <span className="model-tag">Embeddings</span>
                  <span className="model-tag">Search</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Configuration Section */}
        <section className="db-section">
          <div className="section-header">
            <h2 className="section-title">System Configuration</h2>
          </div>

          <div className="config-grid">
            <div className="config-item">
              <div className="config-icon">
                <HiCog />
              </div>
              <div className="config-content">
                <div className="config-label">Similarity Metric</div>
                <div className="config-value">{dbInfo?.similarity_metric}</div>
              </div>
            </div>

            <div className="config-item">
              <div className="config-icon">
                <HiCog />
              </div>
              <div className="config-content">
                <div className="config-label">Collection Type</div>
                <div className="config-value">Persistent</div>
              </div>
            </div>

            <div className="config-item">
              <div className="config-icon">
                <HiCog />
              </div>
              <div className="config-content">
                <div className="config-label">Vector Dimensions</div>
                <div className="config-value">768</div>
              </div>
            </div>

            <div className="config-item">
              <div className="config-icon">
                <HiCog />
              </div>
              <div className="config-content">
                <div className="config-label">Default Top-K</div>
                <div className="config-value">5</div>
              </div>
            </div>
          </div>
        </section>

        {/* Storage Info */}
        <section className="db-section">
          <div className="section-header">
            <h2 className="section-title">Storage Details</h2>
          </div>

          <div className="storage-info">
            <div className="storage-item">
              <div className="storage-label">Document Format</div>
              <div className="storage-value">
                <code>Q: {'{question}'}\nA: {'{answer}'}</code>
              </div>
            </div>

            <div className="storage-item">
              <div className="storage-label">Metadata Fields</div>
              <div className="storage-tags">
                <span className="storage-tag">pdf</span>
                <span className="storage-tag">pdf_title</span>
                <span className="storage-tag">laq_num</span>
                <span className="storage-tag">qa_pair_num</span>
                <span className="storage-tag">type</span>
                <span className="storage-tag">question</span>
                <span className="storage-tag">answer</span>
                <span className="storage-tag">minister</span>
                <span className="storage-tag">date</span>
                <span className="storage-tag">attachments</span>
              </div>
            </div>

            <div className="storage-item">
              <div className="storage-label">Document ID Format</div>
              <div className="storage-value">
                <code>{'{pdf_stem}_{laq_number}_qa{index}'}</code>
              </div>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section className="db-section">
          <div className="section-header">
            <h2 className="section-title">Database Features</h2>
          </div>

          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon">✓</div>
              <div className="feature-title">Semantic Search</div>
              <div className="feature-description">
                Vector-based similarity search using cosine distance
              </div>
            </div>

            <div className="feature-card">
              <div className="feature-icon">✓</div>
              <div className="feature-title">Persistent Storage</div>
              <div className="feature-description">
                Data persists across application restarts
              </div>
            </div>

            <div className="feature-card">
              <div className="feature-icon">✓</div>
              <div className="feature-title">Duplicate Detection</div>
              <div className="feature-description">
                Automatic detection and prevention of duplicate entries
              </div>
            </div>

            <div className="feature-card">
              <div className="feature-icon">✓</div>
              <div className="feature-title">Batch Operations</div>
              <div className="feature-description">
                Efficient batch insertion for better performance
              </div>
            </div>

            <div className="feature-card">
              <div className="feature-icon">✓</div>
              <div className="feature-title">Relevance Filtering</div>
              <div className="feature-description">
                Configurable similarity threshold for result quality
              </div>
            </div>

            <div className="feature-card">
              <div className="feature-icon">✓</div>
              <div className="feature-title">Local Processing</div>
              <div className="feature-description">
                100% local, no external API calls or internet required
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}

export default Database
