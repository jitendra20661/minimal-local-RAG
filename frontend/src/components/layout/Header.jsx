/**
 * Header component
 */

import { useLocation } from 'react-router-dom'
import './Header.css'

function Header() {
  const location = useLocation()

  const getPageTitle = () => {
    const titles = {
      '/': 'LAQ RAG Dashboard',
      '/search': 'Search LAQs',
      '/chat': 'Chat with LAQs',
      '/upload': 'Upload PDF',
      '/database': 'Database Info',
    }
    return titles[location.pathname] || 'LAQ RAG Dashboard'
  }

  return (
    <header className="header">
      <h1 className="header-title">{getPageTitle()}</h1>
      <div className="header-actions">
        {/* Add header actions as needed */}
      </div>
    </header>
  )
}

export default Header
