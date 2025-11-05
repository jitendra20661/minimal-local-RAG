/**
 * Dashboard page component
 */

import { useState, useEffect } from 'react'
import { HiDocumentText, HiSearch, HiChatAlt2, HiDatabase, HiChip } from 'react-icons/hi'
import { SiOpenai } from 'react-icons/si'
import { getDatabaseInfo } from '../services/api'
import './Dashboard.css'

function Dashboard() {
  const [dbInfo, setDbInfo] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchDbInfo = async () => {
      try {
        const data = await getDatabaseInfo()
        setDbInfo(data)
      } catch (error) {
        console.error('Failed to fetch database info:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchDbInfo()
  }, [])

  return (
    <div className="dashboard">
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon">
            <HiDocumentText />
          </div>
          <div className="stat-label">Total LAQs</div>
          <div className="stat-value">{loading ? '...' : dbInfo?.total_documents || 0}</div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">
            <HiChip />
          </div>
          <div className="stat-label">Embedding Model</div>
          <div className="stat-value">{loading ? '...' : dbInfo?.embedding_model || 'N/A'}</div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">
            <SiOpenai />
          </div>
          <div className="stat-label">LLM Model</div>
          <div className="stat-value">{loading ? '...' : dbInfo?.llm_model || 'N/A'}</div>
        </div>

        <div className="stat-card">
          <div className="stat-icon">
            <HiDatabase />
          </div>
          <div className="stat-label">Database</div>
          <div className="stat-value">{loading ? '...' : dbInfo?.collection_name || 'N/A'}</div>
        </div>
      </div>

      <section className="section">
        <div className="section-header">
          <h2 className="section-title">Quick Actions</h2>
        </div>
        <div className="quick-actions">
          <a href="/upload" className="action-card">
            <div className="action-icon">
              <HiDocumentText />
            </div>
            <div className="action-title">Upload PDF</div>
            <div className="action-description">Process new LAQ documents</div>
          </a>

          <a href="/search" className="action-card">
            <div className="action-icon">
              <HiSearch />
            </div>
            <div className="action-title">Search</div>
            <div className="action-description">Find relevant LAQs</div>
          </a>

          <a href="/chat" className="action-card">
            <div className="action-icon">
              <HiChatAlt2 />
            </div>
            <div className="action-title">Chat</div>
            <div className="action-description">Ask questions about LAQs</div>
          </a>
        </div>
      </section>
    </div>
  )
}

export default Dashboard
