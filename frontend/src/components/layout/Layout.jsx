/**
 * Main layout component with sidebar and header
 */

import { useState } from 'react'
import Sidebar from './Sidebar'
import Header from './Header'
import './Layout.css'

function Layout({ children }) {
  return (
    <div className="app-container">
      <Sidebar />
      <main className="main-content">
        <Header />
        <div className="content">
          {children}
        </div>
      </main>
    </div>
  )
}

export default Layout
