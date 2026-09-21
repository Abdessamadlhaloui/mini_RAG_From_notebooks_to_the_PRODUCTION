export function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export function truncate(value, limit = 120) {
  const text = String(value ?? '');
  if (text.length <= limit) {
    return text;
  }
  return `${text.slice(0, Math.max(limit - 1, 0)).trimEnd()}…`;
}

export function shortId(value) {
  const text = String(value ?? '');
  return text.length > 8 ? `${text.slice(0, 4)}…${text.slice(-4)}` : text;
}

export function formatNumber(value, digits = 0) {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return '—';
  }
  return num.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function formatPercent(value, digits = 1) {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return '—';
  }
  return `${(num * 100).toFixed(digits)}%`;
}

export function formatCurrency(value, digits = 4) {
  const num = Number(value);
  if (!Number.isFinite(num)) {
    return '—';
  }
  return new Intl.NumberFormat(undefined, {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(num);
}

export function humanFileSize(bytes) {
  const value = Number(bytes);
  if (!Number.isFinite(value) || value < 0) {
    return '—';
  }
  if (value < 1024) {
    return `${value} B`;
  }
  const units = ['KB', 'MB', 'GB', 'TB'];
  let idx = -1;
  let size = value;
  while (size >= 1024 && idx < units.length - 1) {
    size /= 1024;
    idx += 1;
  }
  return `${size.toFixed(size >= 10 ? 1 : 2)} ${units[Math.max(idx, 0)]}`;
}

export function formatDateTime(value) {
  if (!value) {
    return '—';
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(date);
}

export function formatRelativeTime(value) {
  if (!value) {
    return '—';
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  const delta = date.getTime() - Date.now();
  const seconds = Math.round(Math.abs(delta) / 1000);
  const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: 'auto' });
  if (seconds < 60) return rtf.format(Math.sign(delta) * seconds || 0, 'second');
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return rtf.format(Math.sign(delta) * minutes || 0, 'minute');
  const hours = Math.round(minutes / 60);
  if (hours < 24) return rtf.format(Math.sign(delta) * hours || 0, 'hour');
  const days = Math.round(hours / 24);
  if (days < 30) return rtf.format(Math.sign(delta) * days || 0, 'day');
  const months = Math.round(days / 30);
  if (months < 12) return rtf.format(Math.sign(delta) * months || 0, 'month');
  const years = Math.round(months / 12);
  return rtf.format(Math.sign(delta) * years || 0, 'year');
}

function sanitizeUrl(url) {
  const value = String(url ?? '').trim();
  if (/^(https?:|mailto:)/i.test(value)) {
    return value;
  }
  return '#';
}

function inlineMarkdown(text) {
  let output = escapeHtml(text);
  output = output.replace(/`([^`]+)`/g, '<code>$1</code>');
  output = output.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  output = output.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, label, url) => {
    const safe = sanitizeUrl(url);
    return `<a href="${escapeHtml(safe)}" target="_blank" rel="noreferrer noopener">${label}</a>`;
  });
  return output;
}

function parseTable(lines) {
  if (lines.length < 2) {
    return null;
  }
  const headers = lines[0]
    .split('|')
    .map((cell) => cell.trim())
    .filter(Boolean);
  const separator = lines[1].split('|').every((cell) => /^:?-{3,}:?$/.test(cell.trim()) || !cell.trim());
  if (!headers.length || !separator) {
    return null;
  }
  const rows = lines.slice(2).map((line) =>
    line
      .split('|')
      .map((cell) => cell.trim())
      .filter(Boolean),
  );
  const body = rows
    .filter((row) => row.length)
    .map(
      (row) =>
        `<tr>${headers
          .map((_, index) => `<td>${inlineMarkdown(row[index] || '')}</td>`)
          .join('')}</tr>`,
    )
    .join('');
  const head = headers.map((cell) => `<th>${inlineMarkdown(cell)}</th>`).join('');
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function parseList(lines) {
  const items = lines
    .map((line) => line.replace(/^[-*]\s+/, '').trim())
    .filter(Boolean)
    .map((item) => `<li>${inlineMarkdown(item)}</li>`)
    .join('');
  return `<ul>${items}</ul>`;
}

export function markdownToHtml(markdown) {
  const input = String(markdown ?? '').replace(/\r\n/g, '\n').trim();
  if (!input) {
    return '';
  }

  const blocks = [];
  const parts = input.split(/\n{2,}/);

  for (const part of parts) {
    const trimmed = part.trim();
    if (!trimmed) {
      continue;
    }

    const fenceMatch = trimmed.match(/^```(\w+)?\n([\s\S]*?)```$/);
    if (fenceMatch) {
      const [, language = '', code] = fenceMatch;
      blocks.push(
        `<pre data-language="${escapeHtml(language)}"><code>${escapeHtml(code)}</code></pre>`,
      );
      continue;
    }

    const lines = trimmed.split('\n');
    const first = lines[0].trim();

    const table = parseTable(lines);
    if (table) {
      blocks.push(table);
      continue;
    }

    if (lines.every((line) => /^[-*]\s+/.test(line.trim()))) {
      blocks.push(parseList(lines));
      continue;
    }

    if (first.startsWith('>')) {
      const quoted = lines.map((line) => line.replace(/^>\s?/, '')).join('<br>');
      blocks.push(`<blockquote>${inlineMarkdown(quoted)}</blockquote>`);
      continue;
    }

    if (/^###\s+/.test(first)) {
      blocks.push(`<h3>${inlineMarkdown(first.replace(/^###\s+/, ''))}</h3>`);
      continue;
    }

    if (/^##\s+/.test(first)) {
      blocks.push(`<h2>${inlineMarkdown(first.replace(/^##\s+/, ''))}</h2>`);
      continue;
    }

    if (/^#\s+/.test(first)) {
      blocks.push(`<h1>${inlineMarkdown(first.replace(/^#\s+/, ''))}</h1>`);
      continue;
    }

    blocks.push(`<p>${inlineMarkdown(trimmed).replace(/\n/g, '<br>')}</p>`);
  }

  return blocks.join('');
}

export function sparklineSvg(values, options = {}) {
  const data = Array.from(values || []).map((value) => Number(value)).filter((value) => Number.isFinite(value));
  const width = options.width || 220;
  const height = options.height || 70;
  const stroke = options.stroke || 'var(--accent)';
  const fill = options.fill || 'rgba(139, 92, 246, 0.12)';
  if (!data.length) {
    return `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-hidden="true"><path d="M 0 ${height / 2} L ${width} ${height / 2}" stroke="rgba(148, 163, 184, 0.18)" stroke-width="2" fill="none" /></svg>`;
  }

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const step = width / Math.max(data.length - 1, 1);

  const points = data.map((value, index) => {
    const x = index * step;
    const y = height - ((value - min) / range) * (height - 8) - 4;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  });

  const start = `0,${height}`;
  const end = `${width},${height}`;
  const area = `M ${start} L ${points.join(' L ')} L ${end} Z`;
  const line = `M ${points.join(' L ')}`;

  return `
    <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-hidden="true">
      <path d="${area}" fill="${fill}" />
      <path d="${line}" fill="none" stroke="${stroke}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />
    </svg>
  `;
}

export async function copyText(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(String(text ?? ''));
    return true;
  }
  const textarea = document.createElement('textarea');
  textarea.value = String(text ?? '');
  textarea.style.position = 'fixed';
  textarea.style.left = '-9999px';
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand('copy');
  textarea.remove();
  return true;
}

export function downloadText(filename, content, mimeType = 'text/plain;charset=utf-8') {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
