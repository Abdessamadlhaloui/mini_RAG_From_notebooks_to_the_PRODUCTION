export class ApiError extends Error {
  constructor(status, message, payload = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.payload = payload;
  }
}

export class ApiClient {
  constructor(baseUrl = window.location.origin) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.apiKey = localStorage.getItem('rag_api_key') || '';
  }

  setApiKey(apiKey) {
    this.apiKey = (apiKey || '').trim();
    if (this.apiKey) {
      localStorage.setItem('rag_api_key', this.apiKey);
    } else {
      localStorage.removeItem('rag_api_key');
    }
  }

  headers(extra = {}) {
    const headers = { ...extra };
    if (this.apiKey) {
      headers.Authorization = `Bearer ${this.apiKey}`;
    }
    return headers;
  }

  async request(path, options = {}) {
    const {
      method = 'GET',
      body,
      json = true,
      signal,
      headers = {},
    } = options;

    const requestHeaders = this.headers({ ...headers });
    if (json && body !== undefined && !(body instanceof FormData)) {
      requestHeaders['Content-Type'] = 'application/json';
    }

    const response = await fetch(`${this.baseUrl}${path}`, {
      method,
      headers: requestHeaders,
      body: body instanceof FormData ? body : body === undefined ? undefined : json ? JSON.stringify(body) : body,
      signal,
    });

    if (!response.ok) {
      let payload = null;
      let message = `Request failed with status ${response.status}`;
      try {
        payload = await response.json();
        message = payload?.detail || payload?.message || message;
      } catch {
        try {
          message = await response.text();
        } catch {
          message = message;
        }
      }
      throw new ApiError(response.status, message, payload);
    }

    if (response.status === 204) {
      return null;
    }

    if (!json) {
      return response.text();
    }

    return response.json();
  }

  health() {
    return this.request('/health', { json: true });
  }

  overview() {
    return this.request('/api/v1/overview');
  }

  conversations(page = 1, limit = 50) {
    return this.request(`/api/v1/conversations?page=${page}&limit=${limit}`);
  }

  conversationDetail(conversationId) {
    return this.request(`/api/v1/conversations/${encodeURIComponent(conversationId)}`);
  }

  createConversation(title = '') {
    return this.request('/api/v1/conversations', {
      method: 'POST',
      body: title ? { title } : {},
    });
  }

  updateConversation(conversationId, title) {
    return this.request(`/api/v1/conversations/${encodeURIComponent(conversationId)}`, {
      method: 'PATCH',
      body: { title },
    });
  }

  deleteConversation(conversationId) {
    return this.request(`/api/v1/conversations/${encodeURIComponent(conversationId)}`, {
      method: 'DELETE',
    });
  }

  evaluations(limit = 20) {
    return this.request(`/api/v1/evaluations?limit=${limit}`);
  }

  metricsSnapshot() {
    return this.request('/api/v1/metrics/snapshot');
  }

  query(payload) {
    return this.request('/api/v1/query', {
      method: 'POST',
      body: payload,
    });
  }

  async *streamQuery(payload) {
    const response = await fetch(`${this.baseUrl}/api/v1/query/stream`, {
      method: 'POST',
      headers: this.headers({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(payload),
    });

    if (!response.ok || !response.body) {
      const text = await response.text();
      throw new ApiError(response.status, text || `Streaming request failed (${response.status})`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        let newlineIndex = buffer.indexOf('\n');
        while (newlineIndex >= 0) {
          const line = buffer.slice(0, newlineIndex).trim();
          buffer = buffer.slice(newlineIndex + 1);
          if (line) {
            yield JSON.parse(line);
          }
          newlineIndex = buffer.indexOf('\n');
        }
      }
      const tail = buffer.trim();
      if (tail) {
        yield JSON.parse(tail);
      }
    } finally {
      reader.releaseLock();
    }
  }

  uploadFile(file, onProgress = null) {
    return new Promise((resolve, reject) => {
      const formData = new FormData();
      formData.append('file', file, file.name);

      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${this.baseUrl}/api/v1/ingest`);
      if (this.apiKey) {
        xhr.setRequestHeader('Authorization', `Bearer ${this.apiKey}`);
      }

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && typeof onProgress === 'function') {
          onProgress(Math.round((event.loaded / event.total) * 100));
        }
      };

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText || '{}'));
          } catch {
            resolve({});
          }
          return;
        }
        let payload = null;
        try {
          payload = JSON.parse(xhr.responseText);
        } catch {
          payload = { detail: xhr.responseText };
        }
        reject(new ApiError(xhr.status, payload?.detail || `Upload failed (${xhr.status})`, payload));
      };

      xhr.onerror = () => reject(new ApiError(0, 'Upload failed due to a network error.'));
      xhr.send(formData);
    });
  }
}

export function parsePrometheusMetrics(text) {
  const metrics = {};
  const lines = String(text || '').split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) {
      continue;
    }
    const match = trimmed.match(/^([^{\s]+)(?:\{.*\})?\s+(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)$/i);
    if (!match) {
      continue;
    }
    const [, name, raw] = match;
    const value = Number(raw);
    if (!Number.isNaN(value)) {
      metrics[name] = value;
    }
  }
  return metrics;
}
