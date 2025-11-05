/**
 * API service layer for communicating with FastAPI backend
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Upload API
export const uploadPDF = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/api/upload/', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

// Search API
export const searchLAQs = async (query, topK = 5, threshold = 0.6) => {
  const response = await api.post('/api/search/', {
    query,
    top_k: topK,
    threshold,
  });

  return response.data;
};

// Chat API
export const chatWithLAQs = async (question, topK = 5) => {
  const response = await api.post('/api/chat/', {
    question,
    top_k: topK,
  });

  return response.data;
};

// Database API
export const getDatabaseInfo = async () => {
  const response = await api.get('/api/database/info');
  return response.data;
};

// Health Check
export const healthCheck = async () => {
  const response = await api.get('/api/health');
  return response.data;
};

export default api;
