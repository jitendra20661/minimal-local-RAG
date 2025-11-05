/**
 * Sidebar navigation component
 */

import { Link, useLocation } from 'react-router-dom'
import { HiHome, HiSearch, HiChatAlt2, HiUpload, HiDatabase } from 'react-icons/hi'
import './Sidebar.css'

const iconMap = {
  home: HiHome,
  search: HiSearch,
  chat: HiChatAlt2,
  upload: HiUpload,
  database: HiDatabase,
}

function Sidebar() {
  const location = useLocation()

  const navItems = [
    { path: '/', label: 'Home', icon: 'home' },
    { path: '/search', label: 'Search', icon: 'search' },
    { path: '/chat', label: 'Chat', icon: 'chat' },
    { path: '/upload', label: 'Upload', icon: 'upload' },
    { path: '/database', label: 'Database', icon: 'database' },
  ]

  return (
    <aside className="sidebar">
      {/* <div className="logo">
        <svg viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M20 5L5 12.5V27.5L20 35L35 27.5V12.5L20 5Z" stroke="currentColor" strokeWidth="2" fill="none"/>
          <circle cx="20" cy="20" r="5" fill="#00d4c4"/>
        </svg>
      </div> */}

      <nav className="nav-items">
        {navItems.map((item) => {
          const IconComponent = iconMap[item.icon]
          return (
            <Link
              key={item.path}
              to={item.path}
              className={`nav-item ${location.pathname === item.path ? 'active' : ''}`}
            >
              <IconComponent className="nav-item-icon" />
              <span className="nav-item-label">{item.label}</span>
            </Link>
          )
        })}
      </nav>

      {/* <div className="account-section">
        <div className="account-avatar">
          R
          <span className="pro-badge">PRO</span>
        </div>
      </div> */}
    </aside>
  )
}

export default Sidebar
