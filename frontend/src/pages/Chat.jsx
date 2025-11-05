/**
 * Chat page component
 */

import { useState, useRef, useEffect } from 'react'
import { HiPaperAirplane, HiUser, HiChatAlt2, HiLightningBolt, HiTag, HiCalendar, HiUserCircle } from 'react-icons/hi'
import { SiOpenai } from 'react-icons/si'
import { chatWithLAQs } from '../services/api'
import './Chat.css'

function Chat() {
  const [messages, setMessages] = useState([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  // Auto-scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!inputValue.trim() || isLoading) return

    const userMessage = inputValue.trim()
    setInputValue('')
    setError(null)

    // Add user message to chat
    const newUserMessage = {
      id: Date.now(),
      type: 'user',
      content: userMessage,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, newUserMessage])
    setIsLoading(true)

    try {
      // Call chat API
      const response = await chatWithLAQs(userMessage)

      // Add AI response to chat
      const aiMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        content: response.answer,
        sources: response.sources,
        timestamp: new Date()
      }

      setMessages(prev => [...prev, aiMessage])
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Failed to get response')

      // Add error message to chat
      const errorMessage = {
        id: Date.now() + 1,
        type: 'error',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date()
      }

      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const clearChat = () => {
    setMessages([])
    setError(null)
  }

  const getSimilarityColor = (score) => {
    if (score >= 0.8) return 'high'
    if (score >= 0.6) return 'medium'
    return 'low'
  }

  return (
    <div className="chat-page">
      <div className="chat-container">
        {/* Chat Header */}
        <div className="chat-header">
          <div className="chat-header-info">
            <div className="chat-header-icon">
              <HiChatAlt2 />
            </div>
            <div className="chat-header-text">
              <h2 className="chat-title">Chat with LAQs</h2>
              <p className="chat-subtitle">Ask questions about Legislative Assembly Questions</p>
            </div>
          </div>
          {messages.length > 0 && (
            <button className="clear-chat-button" onClick={clearChat}>
              Clear Chat
            </button>
          )}
        </div>

        {/* Messages Area */}
        <div className="chat-messages">
          {messages.length === 0 ? (
            <div className="chat-empty-state">
              <div className="empty-state-icon">
                <HiChatAlt2 />
              </div>
              <div className="empty-state-title">Start a conversation</div>
              <div className="empty-state-description">
                Ask questions about LAQs and get AI-powered answers with source citations
              </div>
              <div className="suggested-questions">
                <div className="suggested-title">Try asking:</div>
                <button
                  className="suggested-question"
                  onClick={() => setInputValue("What is the education budget?")}
                >
                  <HiLightningBolt />
                  <span>What is the education budget?</span>
                </button>
                <button
                  className="suggested-question"
                  onClick={() => setInputValue("Tell me about healthcare policies")}
                >
                  <HiLightningBolt />
                  <span>Tell me about healthcare policies</span>
                </button>
                <button
                  className="suggested-question"
                  onClick={() => setInputValue("What are the infrastructure plans?")}
                >
                  <HiLightningBolt />
                  <span>What are the infrastructure plans?</span>
                </button>
              </div>
            </div>
          ) : (
            <>
              {messages.map((message) => (
                <div key={message.id} className={`message message-${message.type}`}>
                  <div className="message-avatar">
                    {message.type === 'user' ? (
                      <HiUserCircle />
                    ) : message.type === 'error' ? (
                      <span className="error-icon">!</span>
                    ) : (
                      <SiOpenai />
                    )}
                  </div>

                  <div className="message-content">
                    <div className="message-header">
                      <span className="message-role">
                        {message.type === 'user' ? 'You' : message.type === 'error' ? 'Error' : 'Assistant'}
                      </span>
                      <span className="message-time">
                        {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </span>
                    </div>

                    <div className="message-text">
                      {message.content}
                    </div>

                    {/* Show sources for assistant messages */}
                    {message.type === 'assistant' && message.sources && message.sources.length > 0 && (
                      <div className="message-sources">
                        <div className="sources-header">
                          <HiTag />
                          <span>Sources ({message.sources.length})</span>
                        </div>
                        <div className="sources-list">
                          {message.sources.map((source, index) => (
                            <div key={index} className="source-card">
                              <div className="source-header">
                                <span className="source-laq">LAQ {source.metadata.laq_num}</span>
                                <span className={`source-score score-${getSimilarityColor(source.similarity_score)}`}>
                                  {(source.similarity_score * 100).toFixed(0)}%
                                </span>
                              </div>
                              <div className="source-question">
                                <strong>Q:</strong> {source.question}
                              </div>
                              <div className="source-metadata">
                                <span>
                                  <HiUser />
                                  {source.metadata.minister}
                                </span>
                                <span>
                                  <HiCalendar />
                                  {source.metadata.date}
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {/* Loading Indicator */}
              {isLoading && (
                <div className="message message-assistant">
                  <div className="message-avatar">
                    <SiOpenai />
                  </div>
                  <div className="message-content">
                    <div className="message-header">
                      <span className="message-role">Assistant</span>
                    </div>
                    <div className="typing-indicator">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input Area */}
        <div className="chat-input-container">
          {error && (
            <div className="chat-error">
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="chat-input-form">
            <textarea
              ref={inputRef}
              className="chat-input"
              placeholder="Ask a question about LAQs..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              rows={1}
              disabled={isLoading}
            />
            <button
              type="submit"
              className="chat-send-button"
              disabled={!inputValue.trim() || isLoading}
            >
              <HiPaperAirplane />
            </button>
          </form>

          <div className="chat-input-hint">
            Press <kbd>Enter</kbd> to send, <kbd>Shift + Enter</kbd> for new line
          </div>
        </div>
      </div>
    </div>
  )
}

export default Chat
