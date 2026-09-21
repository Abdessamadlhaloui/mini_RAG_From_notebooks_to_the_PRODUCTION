import { ApiClient, ApiError, parsePrometheusMetrics } from './api.js';
import {
  copyText,
  downloadText,
  escapeHtml,
  formatCurrency,
  formatDateTime,
  formatNumber,
  formatPercent,
  formatRelativeTime,
  humanFileSize,
  markdownToHtml,
  shortId,
  sparklineSvg,
  truncate,
} from './ui.js';

const PLACEHOLDERS = [
  'Ask anything...',
  'Search documents...',
  'Upload a PDF...',
  'Summarize this report...',
  'Explain this chart...',
  'Use / for commands...',
];

const QUICK_PROMPTS = [
  'Summarize the latest upload.',
  'Find duplicate files and explain the overlap.',
  'Show the key risks in this workspace.',
  'Explain the most recent evaluation score.',
  'Generate a concise summary with citations.',
];

const COMMANDS = [
  { label: '/summarize', prompt: 'Summarize the selected document in 5 bullets with citations.' },
  { label: '/search', prompt: 'Search the workspace for the most relevant documents about this topic.' },
  { label: '/explain', prompt: 'Explain this answer in plain English with evidence.' },
  { label: '/compare', prompt: 'Compare the two most similar documents and highlight differences.' },
  { label: '/export', prompt: 'Export this conversation as Markdown with citations.' },
];

const DEFAULT_OVERVIEW = {
  version: 'dev',
  counts: {
    documents: 0,
    conversations: 0,
    messages: 0,
    memories: 0,
    evaluations: 0,
  },
  feature_flags: {},
  recent_documents: [],
  recent_conversations: [],
  recent_evaluations: [],
  metrics: '',
};

const state = {
  themeMode: localStorage.getItem('rag_theme_mode') || 'system',
  activeSection: location.hash.replace('#', '') || 'dashboard',
  search: '',
  draft: '',
  apiKey: localStorage.getItem('rag_api_key') || '',
  overview: DEFAULT_OVERVIEW,
  conversations: [],
  conversationDetail: null,
  activeConversationId: localStorage.getItem('rag_active_conversation') || '',
  selectedDocument: null,
  selectedSource: null,
  selectedSourceIndex: -1,
  selectedEvaluation: null,
  parsedMetrics: {},
  metricsText: '',
  authState: 'unknown',
  upload: {
    active: false,
    filename: '',
    progress: 0,
    stage: 'Idle',
    status: 'idle',
    error: '',
  },
  rightCollapsed: localStorage.getItem('rag_right_collapsed') === '1',
  rightWidth: Number(localStorage.getItem('rag_right_width') || 390),
  draftBoard: localStorage.getItem('rag_draft_board') || '',
  pinnedConversationIds: JSON.parse(localStorage.getItem('rag_pinned_conversations') || '[]'),
  loading: true,
  loadingConversation: false,
  streaming: false,
  streamBuffer: '',
  streamMetadata: null,
  streamError: '',
};

const api = new ApiClient(window.location.origin);

const els = {
  app: document.getElementById('app'),
  toastHost: document.getElementById('toast-host'),
};

if (!els.app || !els.toastHost) {
  throw new Error('Application shell is missing from index.html');
}

let renderQueued = false;
let placeholderTimer = null;
let currentPlaceholderIndex = 0;
let uploadStageTimer = null;
let uploadMilestones = ['Uploading', 'Extracting', 'Chunking', 'Embedding', 'Indexing'];
let activeVoiceRecognition = null;
let dragResize = null;

function setState(partial, options = {}) {
  Object.assign(state, partial);
  if (!options.silent) {
    scheduleRender();
  }
}

function scheduleRender() {
  if (renderQueued) {
    return;
  }
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    render();
  });
}

function setThemeMode(mode) {
  const resolved = resolveTheme(mode);
  state.themeMode = mode;
  document.documentElement.dataset.theme = resolved;
  localStorage.setItem('rag_theme_mode', mode);
}

function resolveTheme(mode = state.themeMode) {
  if (mode === 'system') {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  return mode === 'light' ? 'light' : 'dark';
}

function applyLayoutPrefs() {
  document.documentElement.style.setProperty('--right-panel-width', `${state.rightWidth}px`);
  document.documentElement.dataset.theme = resolveTheme();
}

function makeInitialShell() {
  els.app.innerHTML = `
    <aside class="panel panel-left">
      <div class="sidebar">
        <div class="workspace-switcher">
          <div class="workspace-switcher__meta">
            <strong>Enterprise RAG</strong>
            <span id="workspace-subtitle">Loading workspace…</span>
          </div>
          <span class="chip chip--accent" id="workspace-environment">dev</span>
        </div>
        <div id="left-panel-body" class="sidebar-scroll"></div>
      </div>
    </aside>
    <section class="panel panel-center">
      <header class="topbar">
        <div class="topbar__title">
          <strong>AI Workspace</strong>
          <span id="topbar-status">Connected workspace shell</span>
        </div>
        <div class="topbar__search" role="search">
          <span class="chip chip--accent">⌘K</span>
          <input id="global-search" type="search" autocomplete="off" spellcheck="false" placeholder="Search conversations, documents, and metrics" />
        </div>
        <div class="topbar__actions">
          <button class="button button--small button--ghost" type="button" data-action="refresh-all">Refresh</button>
          <button class="button button--small button--ghost" type="button" data-action="create-conversation">New chat</button>
          <button class="button button--small button--ghost" type="button" data-action="upload-file">Upload</button>
          <button class="button button--small button--ghost" type="button" data-action="theme-cycle">Theme</button>
          <button class="button button--small button--primary" type="button" data-action="open-settings">Settings</button>
        </div>
      </header>
      <div id="center-content" class="content"></div>
    </section>
    <aside class="panel panel-right ${state.rightCollapsed ? 'is-collapsed' : ''}" id="copilot-panel">
      <button class="right-rail-toggle" type="button" aria-label="Toggle copilot" data-action="toggle-copilot">
        <span class="sr-only">Toggle copilot</span>
        <span aria-hidden="true">${state.rightCollapsed ? '›' : '‹'}</span>
      </button>
      <div class="copilot">
        <div class="copilot__header">
          <div>
            <h2>AI Copilot</h2>
            <p>Persistent context, streaming answers, sources, and citations.</p>
            <div class="copilot__context" id="copilot-context"></div>
          </div>
          <div class="copilot__status">
            <span class="status-dot ${state.streaming ? '' : 'status-dot--warn'}"></span>
            <span id="copilot-status">${state.streaming ? 'Streaming' : 'Ready'}</span>
          </div>
        </div>
        <div class="copilot__body" id="copilot-body"></div>
      </div>
      <div class="resize-handle" data-resize-handle aria-hidden="true"></div>
      <div class="right-collapsed-label">Copilot</div>
    </aside>
  `;
}

function render() {
  const workspace = document.getElementById('workspace-subtitle');
  const environment = document.getElementById('workspace-environment');
  const topbarStatus = document.getElementById('topbar-status');
  const globalSearch = document.getElementById('global-search');
  const leftBody = document.getElementById('left-panel-body');
  const centerContent = document.getElementById('center-content');
  const copilotBody = document.getElementById('copilot-body');
  const copilotContext = document.getElementById('copilot-context');
  const copilotStatus = document.getElementById('copilot-status');
  const copilotPanel = document.getElementById('copilot-panel');

  if (!workspace || !environment || !topbarStatus || !globalSearch || !leftBody || !centerContent || !copilotBody || !copilotContext || !copilotStatus || !copilotPanel) {
    return;
  }

  workspace.textContent = state.overview?.counts?.documents
    ? `${formatNumber(state.overview.counts.documents)} documents indexed`
    : 'Premium AI workspace';
  environment.textContent = state.overview?.version || 'dev';
  topbarStatus.textContent = state.authState === 'ok'
    ? 'Connected to the RAG API'
    : state.authState === 'invalid'
      ? 'API key rejected'
      : state.apiKey
        ? 'Checking API key'
        : 'API key required';

  if (globalSearch.value !== state.search) {
    globalSearch.value = state.search;
  }

  leftBody.innerHTML = renderLeftSidebar();
  centerContent.innerHTML = renderCenterContent();
  copilotBody.innerHTML = renderCopilotPanel();
  copilotContext.innerHTML = renderCopilotContext();
  copilotStatus.textContent = state.streaming ? 'Streaming' : state.loading ? 'Syncing' : 'Ready';

  copilotPanel.classList.toggle('is-collapsed', state.rightCollapsed);
}

function renderLeftSidebar() {
  const pinned = (state.conversations || []).filter((conversation) => state.pinnedConversationIds.includes(conversation.conversation_id));
  const recentDocs = state.overview?.recent_documents || [];
  const nav = [
    ['dashboard', 'Overview', 'Workspace summary and performance'],
    ['documents', 'Documents', 'Upload queue and document preview'],
    ['conversations', 'Conversations', 'Pinned chats and history'],
    ['metrics', 'Metrics', 'Tokens, cost, latency, and evaluations'],
    ['settings', 'Settings', 'Theme, API key, and feature flags'],
  ];

  return `
    <div class="nav">
      <div class="nav__group">
        <div class="nav__heading">Navigation</div>
        ${nav
          .map(
            ([section, label, hint]) => `
              <button class="nav__button ${state.activeSection === section ? 'is-active' : ''}" type="button" data-section="${section}">
                <span class="avatar">${label.charAt(0)}</span>
                <span>
                  <strong>${escapeHtml(label)}</strong>
                  <small>${escapeHtml(hint)}</small>
                </span>
              </button>
            `,
          )
          .join('')}
      </div>
      <div class="divider"></div>
      <div class="nav__group">
        <div class="nav__heading">Pinned Chats</div>
        ${pinned.length ? renderConversationMiniList(pinned, true) : renderEmptyBlock('No pinned chats yet', 'Pin a conversation to keep it in quick reach.')}
      </div>
      <div class="divider"></div>
      <div class="nav__group">
        <div class="nav__heading">Recent Documents</div>
        ${recentDocs.length ? renderDocumentMiniList(recentDocs) : renderEmptyBlock('No uploads yet', 'Drop a PDF, folder, or text file to populate the workspace.')}
      </div>
      <div class="divider"></div>
      <div class="nav__group">
        <div class="nav__heading">Keyboard</div>
        <div class="card card--soft">
          <div class="card__body">
            <div class="list list--dense">
              <div class="list__item"><span class="chip">⌘K</span><span>Focus search</span></div>
              <div class="list__item"><span class="chip">⌘↵</span><span>Send message</span></div>
              <div class="list__item"><span class="chip">/</span><span>Open command prompt</span></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;
}

function renderEmptyBlock(title, text) {
  return `
    <div class="empty-state">
      <h3>${escapeHtml(title)}</h3>
      <p>${escapeHtml(text)}</p>
    </div>
  `;
}

function renderConversationMiniList(conversations, compact = false) {
  return `
    <div class="list ${compact ? 'list--dense' : ''}">
      ${conversations
        .map(
          (conversation) => `
            <button class="list__item clickable ${state.activeConversationId === conversation.conversation_id ? 'list__item--active' : ''}" type="button" data-conversation-id="${escapeHtml(conversation.conversation_id)}">
              <span class="avatar">${escapeHtml((conversation.title || 'C').charAt(0))}</span>
              <span style="min-width:0;">
                <strong>${escapeHtml(conversation.title || 'Untitled')}</strong>
                <small>${formatNumber(conversation.message_count || 0)} messages · ${formatRelativeTime(conversation.updated_at)}</small>
              </span>
            </button>
          `,
        )
        .join('')}
    </div>
  `;
}

function renderDocumentMiniList(documents) {
  return `
    <div class="list list--dense">
      ${documents
        .map(
          (document) => `
            <button class="list__item clickable ${isSelectedDocument(document) ? 'list__item--active' : ''}" type="button" data-document-id="${escapeHtml(document.document_id || document._id || document.filename)}">
              <span class="avatar">D</span>
              <span style="min-width:0;">
                <strong>${escapeHtml(document.filename || 'Untitled')}</strong>
                <small>${escapeHtml(document.status || 'unknown')} · ${formatNumber(document.chunks_processed || 0)} chunks · ${formatRelativeTime(document.timestamp)}</small>
              </span>
            </button>
          `,
        )
        .join('')}
    </div>
  `;
}

function isSelectedDocument(document) {
  const selectedId = state.selectedDocument?.document_id || state.selectedDocument?._id || '';
  const docId = document.document_id || document._id || '';
  return selectedId && selectedId === docId;
}

function renderCenterContent() {
  const section = state.activeSection || 'dashboard';
  return `
    <div class="section ${section === 'dashboard' ? 'is-active' : ''}">${section === 'dashboard' ? renderDashboardSection() : ''}</div>
    <div class="section ${section === 'documents' ? 'is-active' : ''}">${section === 'documents' ? renderDocumentsSection() : ''}</div>
    <div class="section ${section === 'conversations' ? 'is-active' : ''}">${section === 'conversations' ? renderConversationsSection() : ''}</div>
    <div class="section ${section === 'metrics' ? 'is-active' : ''}">${section === 'metrics' ? renderMetricsSection() : ''}</div>
    <div class="section ${section === 'settings' ? 'is-active' : ''}">${section === 'settings' ? renderSettingsSection() : ''}</div>
  `;
}

function renderDashboardSection() {
  const overview = state.overview || DEFAULT_OVERVIEW;
  const counts = overview.counts || DEFAULT_OVERVIEW.counts;
  const metrics = state.parsedMetrics || {};
  const evals = overview.recent_evaluations || [];
  const recentDocuments = overview.recent_documents || [];
  const recentConversations = overview.recent_conversations || [];
  const greeting = makeGreeting();
  const totalCost = metrics.rag_estimated_cost_usd_total || 0;
  const queryTotal = metrics.rag_query_total || 0;
  const cacheHits = metrics.rag_query_cache_hit_total || 0;
  const hitRate = queryTotal ? cacheHits / queryTotal : 0;
  const avgLatency = metrics.rag_query_latency_milliseconds_avg || 0;
  const answerCorrectnessSeries = evals
    .slice()
    .reverse()
    .map((entry) => Number(entry.answer_correctness ?? entry.recall ?? 0) || 0);
  const latencySeries = evals.slice().reverse().map((entry) => Number(entry.latency_ms || 0));
  const hallucinationSeries = evals.slice().reverse().map((entry) => Number(entry.hallucination_rate ?? 0));

  return `
    <div class="section-grid dashboard">
      <div class="section-grid">
        <section class="hero">
          <span class="hero__eyebrow">AI Operating System</span>
          <h1>${escapeHtml(greeting.title)}</h1>
          <p>${escapeHtml(greeting.subtitle)}</p>
          <div class="hero__actions">
            ${QUICK_PROMPTS.slice(0, 3)
              .map((prompt) => `<button class="button button--small button--ghost" type="button" data-prompt="${escapeHtml(prompt)}">${escapeHtml(prompt)}</button>`)
              .join('')}
          </div>
        </section>
        <section class="card">
          <div class="card__header">
            <h2>Workspace summary</h2>
            <span class="chip chip--accent">Updated ${formatRelativeTime(overview.updated_at || new Date())}</span>
          </div>
          <div class="card__body">
            <div class="metric-grid">
              ${renderMetricCard('Documents', counts.documents, `${recentDocuments.length} recent ingests`)}
              ${renderMetricCard('Conversations', counts.conversations, `${formatNumber(counts.messages)} messages stored`)}
              ${renderMetricCard('Evaluations', counts.evaluations, `${formatNumber(counts.memories)} memories retained`)}
              ${renderMetricCard('Cache hit rate', formatPercent(hitRate), `${formatNumber(cacheHits)} hits out of ${formatNumber(queryTotal)} queries`)}
            </div>
          </div>
        </section>
        <div class="grid-auto">
          <section class="sparkline-card">
            <div class="sparkline-card__title">
              <strong>Latency trend</strong>
              <span class="chip">${formatNumber(avgLatency, 1)} ms avg</span>
            </div>
            <div class="sparkline-card__chart">${sparklineSvg(latencySeries, { stroke: 'var(--accent-2)', fill: 'rgba(56, 189, 248, 0.14)' })}</div>
          </section>
          <section class="sparkline-card">
            <div class="sparkline-card__title">
              <strong>Answer quality</strong>
              <span class="chip">${evals.length ? formatNumber((evals.reduce((sum, entry) => sum + Number(entry.answer_correctness || 0), 0) / evals.length) || 0, 2) : '0.00'}</span>
            </div>
            <div class="sparkline-card__chart">${sparklineSvg(answerCorrectnessSeries, { stroke: 'var(--success)', fill: 'rgba(52, 211, 153, 0.12)' })}</div>
          </section>
          <section class="sparkline-card">
            <div class="sparkline-card__title">
              <strong>Hallucination risk</strong>
              <span class="chip">${evals.length ? formatNumber((evals.reduce((sum, entry) => sum + Number(entry.hallucination_rate || 0), 0) / evals.length) || 0, 2) : '0.00'}</span>
            </div>
            <div class="sparkline-card__chart">${sparklineSvg(hallucinationSeries, { stroke: 'var(--danger)', fill: 'rgba(251, 113, 133, 0.12)' })}</div>
          </section>
        </div>
        <section class="card">
          <div class="card__header">
            <h2>Recent documents</h2>
            <button class="button button--small button--ghost" type="button" data-section="documents">Open document studio</button>
          </div>
          <div class="card__body">
            ${recentDocuments.length ? renderDocumentTiles(recentDocuments) : renderEmptyBlock('No documents yet', 'Upload a PDF or TXT file to activate document intelligence.')}
          </div>
        </section>
      </div>
      <div class="section-grid">
        <section class="card card--soft">
          <div class="card__header">
            <h2>Recommended actions</h2>
            <span class="chip chip--accent">Context aware</span>
          </div>
          <div class="card__body">
            ${renderRecommendationCards(overview)}
          </div>
        </section>
        <section class="card">
          <div class="card__header">
            <h2>Working draft</h2>
            <button class="button button--small button--ghost" type="button" data-action="clear-draft">Clear</button>
          </div>
          <div class="card__body">
            <div class="field">
              <label for="draft-board">Capture snippets, exports, or generated text</label>
              <textarea id="draft-board" class="input mono" rows="12" placeholder="Drop answer text here for editing, export, or later insertion.">${escapeHtml(state.draftBoard)}</textarea>
            </div>
            <div class="toolbar mt-md">
              <div class="toolbar__group">
                <button class="button button--small button--ghost" type="button" data-action="copy-draft">Copy draft</button>
                <button class="button button--small button--ghost" type="button" data-action="download-draft-md">Download Markdown</button>
                <button class="button button--small button--ghost" type="button" data-action="download-draft-json">Download JSON</button>
              </div>
              <span class="chip">${formatNumber(state.draftBoard.length)} chars</span>
            </div>
          </div>
        </section>
        <section class="card">
          <div class="card__header">
            <h2>Recent conversations</h2>
            <button class="button button--small button--ghost" type="button" data-section="conversations">Open history</button>
          </div>
          <div class="card__body">
            ${recentConversations.length ? renderConversationMiniList(recentConversations) : renderEmptyBlock('No conversations yet', 'Ask a question to create the first conversation.')}
          </div>
        </section>
        <section class="card">
          <div class="card__header">
            <h2>Platform metrics</h2>
            <span class="chip">${formatCurrency(totalCost)}</span>
          </div>
          <div class="card__body">
            <div class="stat-grid">
              ${renderStat('Queries', formatNumber(queryTotal), `${formatNumber(cacheHits)} cached hits`)}
              ${renderStat('Average latency', `${formatNumber(avgLatency, 1)} ms`, 'Request path latency')}
              ${renderStat('Estimated cost', formatCurrency(totalCost), 'Accumulated generation cost')}
              ${renderStat('Empty retrievals', formatNumber(metrics.rag_query_empty_retrieval_total || 0), 'No-match responses')}
            </div>
          </div>
        </section>
      </div>
    </div>
  `;
}

function renderMetricCard(label, value, detail) {
  return `
    <div class="metric-card">
      <div class="metric-card__label">${escapeHtml(label)}</div>
      <div class="metric-card__value">${value}</div>
      <div class="metric-card__detail">${escapeHtml(detail)}</div>
    </div>
  `;
}

function renderStat(label, value, detail) {
  return `
    <div class="stat">
      <div class="stat__label">${escapeHtml(label)}</div>
      <div class="stat__value">${escapeHtml(String(value))}</div>
      <div class="stat__meta">${escapeHtml(detail)}</div>
    </div>
  `;
}

function renderDocumentTiles(documents) {
  return `
    <div class="grid-auto">
      ${documents
        .slice(0, 6)
        .map((document) => {
          const statusClass = document.status === 'failed' ? 'chip--danger' : 'chip--success';
          return `
            <button class="card clickable" type="button" data-document-id="${escapeHtml(document.document_id || document._id || document.filename)}" style="text-align:left;">
              <div class="card__body">
                <div class="flex-between gap-sm">
                  <strong>${escapeHtml(document.filename || 'Untitled')}</strong>
                  <span class="chip ${statusClass}">${escapeHtml(document.status || 'unknown')}</span>
                </div>
                <p class="muted mt-sm">${truncate(document.error || `${formatNumber(document.chunks_processed || 0)} chunks · ${formatNumber(document.page_count || 0)} pages`, 120)}</p>
                <div class="toolbar mt-md">
                  <span class="chip">${formatRelativeTime(document.timestamp)}</span>
                  <span class="chip">${escapeHtml(document.embedding_version || 'unversioned')}</span>
                </div>
              </div>
            </button>
          `;
        })
        .join('')}
    </div>
  `;
}

function renderRecommendationCards(overview) {
  const docs = overview.recent_documents || [];
  const failures = docs.filter((doc) => doc.status === 'failed');
  const ocr = docs.filter((doc) => doc.has_ocr);
  const prompts = [];
  if (docs.length === 0) {
    prompts.push(['Upload a PDF', 'Drop a document to activate OCR, chunking, embeddings, and citations.']);
  } else {
    prompts.push(['Summarize newest document', 'Generate a concise, citation-backed summary of the latest ingestion.']);
    prompts.push(['Find duplicates', 'Detect near-duplicate uploads and explain overlaps.']);
  }
  if (failures.length) {
    prompts.push(['Fix ingestion errors', `There ${failures.length === 1 ? 'is' : 'are'} ${failures.length} failed upload${failures.length === 1 ? '' : 's'} that need attention.`]);
  }
  if (!ocr.length && docs.length) {
    prompts.push(['Investigate OCR coverage', 'Some documents have no OCR metadata. Check scanned PDFs and image-heavy files.']);
  }

  const alerts = [];
  if (overview.counts?.evaluations === 0) {
    alerts.push(['No evaluation data yet', 'Ask a question to start recording answer quality and latency telemetry.']);
  }
  if ((state.parsedMetrics.rag_query_cache_hit_total || 0) === 0 && (state.parsedMetrics.rag_query_total || 0) > 0) {
    alerts.push(['Cache is cold', 'Repeated queries will start paying off once the response cache is populated.']);
  }

  return `
    <div class="list">
      ${prompts
        .map(
          ([title, description]) => `
            <button class="list__item clickable" type="button" data-prompt="${escapeHtml(title)}">
              <span class="avatar">A</span>
              <span style="min-width:0;">
                <strong>${escapeHtml(title)}</strong>
                <small>${escapeHtml(description)}</small>
              </span>
            </button>
          `,
        )
        .join('')}
    </div>
    <div class="divider"></div>
    <div class="list">
      ${alerts
        .map(
          ([title, description]) => `
            <div class="list__item">
              <span class="avatar">!</span>
              <span style="min-width:0;">
                <strong>${escapeHtml(title)}</strong>
                <small>${escapeHtml(description)}</small>
              </span>
            </div>
          `,
        )
        .join('')}
    </div>
  `;
}

function renderDocumentsSection() {
  const docs = state.overview?.recent_documents || [];
  const selected = state.selectedDocument || docs[0] || null;
  return `
    <div class="section-grid dual">
      <section class="card">
        <div class="card__header">
          <h2>Document upload studio</h2>
          <span class="chip chip--accent">Drag, drop, or paste</span>
        </div>
        <div class="card__body">
          <div class="upload-zone ${state.upload.active ? 'is-dragging' : ''}" id="upload-zone" tabindex="0" role="button" aria-label="Upload documents">
            <div class="upload-zone__meta">
              <span class="hero__eyebrow">${state.upload.status === 'idle' ? 'Ingestion' : escapeHtml(state.upload.stage)}</span>
              <h3 class="mt-0">${state.upload.filename ? escapeHtml(state.upload.filename) : 'Drop one or more PDF, TXT, or image files here'}</h3>
              <p class="muted">Uploads support OCR, layout parsing, table extraction, image captioning, formula recognition, and semantic chunking.</p>
            </div>
            <div class="progress ${state.upload.status === 'processing' ? 'progress--indeterminate' : ''}">
              <span style="width: ${Math.max(4, state.upload.progress || 0)}%;"></span>
            </div>
            <div class="toolbar">
              <div class="toolbar__group">
                <button class="button button--small button--primary" type="button" data-action="upload-file">Choose files</button>
                <button class="button button--small button--ghost" type="button" data-action="upload-folder">Upload folder</button>
                <button class="button button--small button--ghost" type="button" data-action="paste-upload">Paste files</button>
              </div>
              <span class="chip">${state.upload.stage}</span>
            </div>
            ${state.upload.error ? `<div class="chip chip--danger">${escapeHtml(state.upload.error)}</div>` : ''}
          </div>
          <div class="mt-lg">
            <div class="toolbar">
              <h3 class="mt-0">Recent ingests</h3>
              <span class="chip">${formatNumber(docs.length)} files tracked</span>
            </div>
            ${docs.length ? renderDocumentTiles(docs) : renderEmptyBlock('No documents yet', 'This workspace will populate after the first ingest.')}
          </div>
        </div>
      </section>
      <section class="card">
        <div class="card__header">
          <h2>Document preview</h2>
          <span class="chip">${selected ? escapeHtml(selected.status || 'unknown') : 'No selection'}</span>
        </div>
        <div class="card__body">
          ${selected ? renderDocumentPreview(selected) : renderEmptyDocumentPreview()}
        </div>
      </section>
    </div>
  `;
}

function renderEmptyDocumentPreview() {
  return `
    <div class="pdf-placeholder">
      <div class="pdf-placeholder__inner">
        <span class="hero__eyebrow">Preview</span>
        <h3>Select a document</h3>
        <p>Choose a recent ingest or a source citation to inspect the extracted metadata and content preview.</p>
      </div>
    </div>
  `;
}

function renderDocumentPreview(document) {
  const selected = document || {};
  const metadataRows = [
    ['Document ID', selected.document_id || selected._id || '—'],
    ['Filename', selected.filename || '—'],
    ['Status', selected.status || '—'],
    ['File type', selected.file_type || '—'],
    ['Chunks', selected.chunks_processed ?? 0],
    ['Pages', selected.page_count ?? '—'],
    ['OCR', selected.has_ocr ? 'Yes' : 'No'],
    ['OCR confidence', selected.ocr_confidence != null ? formatNumber(selected.ocr_confidence, 2) : '—'],
    ['Tables', selected.tables_count ?? 0],
    ['Images', selected.images_count ?? 0],
    ['Formulas', selected.formulas_count ?? 0],
    ['Embedding version', selected.embedding_version || '—'],
    ['Timestamp', formatDateTime(selected.timestamp)],
  ];

  return `
    <div class="section-grid">
      <div class="pdf-placeholder">
        <div class="pdf-placeholder__inner">
          <span class="hero__eyebrow">${escapeHtml(selected.status || 'document')}</span>
          <h3>${escapeHtml(selected.filename || 'Untitled document')}</h3>
          <p>${escapeHtml(selected.error || 'Document metadata and extraction summary. A live PDF viewer can be attached when the backend exposes source file URLs.')}</p>
        </div>
      </div>
      <div class="card card--soft">
        <div class="card__body">
          <div class="stat-grid">
            ${metadataRows
              .map(
                ([label, value]) => `
                  <div class="stat">
                    <div class="stat__label">${escapeHtml(label)}</div>
                    <div class="stat__value">${escapeHtml(String(value))}</div>
                  </div>
                `,
              )
              .join('')}
          </div>
        </div>
      </div>
    </div>
  `;
}

function renderConversationsSection() {
  const conversations = filterConversations(state.conversations);
  const detail = state.conversationDetail;
  const messages = detail?.messages || [];
  return `
    <div class="section-grid dual">
      <section class="card">
        <div class="card__header">
          <h2>Conversation history</h2>
          <span class="chip">${formatNumber(conversations.length)} visible</span>
        </div>
        <div class="card__body">
          <div class="field mb-md">
            <label for="conversation-search">Filter history</label>
            <input id="conversation-search" class="input" type="search" value="${escapeHtml(state.search)}" placeholder="Search chats, documents, and tags" />
          </div>
          ${conversations.length ? renderConversationMiniList(conversations) : renderEmptyBlock('No matching conversations', 'Try another search or start a fresh conversation.')}
        </div>
      </section>
      <section class="card">
        <div class="card__header">
          <h2>${escapeHtml(detail?.title || 'Conversation detail')}</h2>
          <div class="toolbar__group">
            ${detail ? `<button class="button button--small button--ghost" type="button" data-action="rename-conversation">Rename</button>` : ''}
            ${detail ? `<button class="button button--small button--ghost" type="button" data-action="pin-conversation">${state.pinnedConversationIds.includes(detail.conversation_id) ? 'Unpin' : 'Pin'}</button>` : ''}
            ${detail ? `<button class="button button--small button--ghost" type="button" data-action="export-conversation">Export</button>` : ''}
            ${detail ? `<button class="button button--small button--danger" type="button" data-action="delete-conversation">Delete</button>` : ''}
          </div>
        </div>
        <div class="card__body">
          ${detail ? renderConversationDetail(detail, messages) : renderEmptyConversationDetail()}
        </div>
      </section>
    </div>
  `;
}

function renderEmptyConversationDetail() {
  return `
    <div class="empty-state">
      <h3>No conversation selected</h3>
      <p>Ask a question in the copilot or choose a conversation from the history list to inspect its messages, sources, and citations.</p>
    </div>
  `;
}

function renderConversationDetail(detail, messages) {
  return `
    <div class="timeline">
      <div class="toolbar">
        <div class="toolbar__group">
          <span class="chip chip--accent">${formatNumber(detail.message_count || messages.length)} messages</span>
          <span class="chip">${formatRelativeTime(detail.updated_at)}</span>
          <span class="chip">ID ${shortId(detail.conversation_id)}</span>
        </div>
      </div>
      ${messages.length ? messages.map((message, index) => renderConversationMessage(message, index, messages)).join('') : renderEmptyBlock('Empty history', 'The conversation exists but has no stored messages yet.')}
    </div>
  `;
}

function renderConversationMessage(message, index, messages) {
  const isAssistant = message.role === 'assistant';
  const roleLabel = isAssistant ? 'Assistant' : message.role === 'user' ? 'User' : message.role;
  const previousUser = [...messages.slice(0, index)].reverse().find((item) => item.role === 'user');
  const sourceCards = Array.isArray(message.sources) && message.sources.length
    ? `
      <div class="message__sources">
        ${message.sources
          .map(
            (source, sourceIndex) => `
              <button class="source-card clickable ${state.selectedSourceIndex === sourceIndex && state.selectedSource?.content === source.content ? 'is-selected' : ''}" type="button" data-source-index="${sourceIndex}" data-source-json="${escapeHtml(JSON.stringify(source))}">
                <div class="source-card__top">
                  <span>${escapeHtml(source.metadata?.source_filename || source.metadata?.filename || 'Source')}</span>
                  <span>${source.score != null ? `score ${formatNumber(source.score, 3)}` : 'source'}</span>
                </div>
                <div class="source-card__excerpt">${escapeHtml(truncate(source.content || source.page_content || '', 280))}</div>
              </button>
            `,
          )
          .join('')}
      </div>
    `
    : '';

  return `
    <article class="message ${isAssistant ? 'message--assistant' : 'message--user'}">
      <div class="message__meta">
        <strong>${escapeHtml(roleLabel)}</strong>
        <div class="toolbar__group">
          <span class="chip">${formatDateTime(message.created_at)}</span>
          ${message.tokens_used != null ? `<span class="chip">${formatNumber(message.tokens_used)} tokens</span>` : ''}
        </div>
      </div>
      <div class="message__body">${markdownToHtml(message.content || '')}</div>
      ${sourceCards}
      ${isAssistant ? `
        <div class="message__actions">
          <button class="button button--small button--ghost" type="button" data-action="copy-message" data-content="${escapeHtml(message.content || '')}">Copy</button>
          <button class="button button--small button--ghost" type="button" data-action="insert-message" data-content="${escapeHtml(message.content || '')}">Insert</button>
          <button class="button button--small button--ghost" type="button" data-action="download-message" data-content="${escapeHtml(message.content || '')}" data-role="${escapeHtml(roleLabel)}">Download</button>
          ${previousUser ? `<button class="button button--small button--primary" type="button" data-action="retry-message" data-prompt="${escapeHtml(previousUser.content || '')}">Retry</button>` : ''}
          ${previousUser ? `<button class="button button--small button--ghost" type="button" data-action="explain-message" data-prompt="${escapeHtml(previousUser.content || '')}">Explain</button>` : ''}
        </div>
      ` : ''}
    </article>
  `;
}

function renderMetricsSection() {
  const metrics = state.parsedMetrics || {};
  const evals = state.overview?.recent_evaluations || [];
  const latencySeries = evals.slice().reverse().map((entry) => Number(entry.latency_ms || 0));
  const correctnessSeries = evals.slice().reverse().map((entry) => Number(entry.answer_correctness || 0));
  const rows = [
    ['Queries', metrics.rag_query_total || 0],
    ['Cache hits', metrics.rag_query_cache_hit_total || 0],
    ['Empty retrievals', metrics.rag_query_empty_retrieval_total || 0],
    ['Errors', metrics.rag_query_error_total || 0],
    ['Tokens', metrics.rag_tokens_total || 0],
    ['Cost', formatCurrency(metrics.rag_estimated_cost_usd_total || 0)],
    ['Latency avg', `${formatNumber(metrics.rag_query_latency_milliseconds_avg || 0, 1)} ms`],
    ['Latency count', metrics.rag_query_latency_milliseconds_count || 0],
  ];

  return `
    <div class="section-grid">
      <section class="card">
        <div class="card__header">
          <h2>Telemetry dashboard</h2>
          <span class="chip chip--accent">Prometheus snapshot</span>
        </div>
        <div class="card__body">
          <div class="metric-grid">
            ${rows
              .map(
                ([label, value]) => `
                  <div class="metric-card">
                    <div class="metric-card__label">${escapeHtml(label)}</div>
                    <div class="metric-card__value">${escapeHtml(String(value))}</div>
                  </div>
                `,
              )
              .join('')}
          </div>
        </div>
      </section>
      <div class="grid-auto">
        <section class="sparkline-card">
          <div class="sparkline-card__title">
            <strong>Latency</strong>
            <span class="chip">${formatNumber(metrics.rag_query_latency_milliseconds_avg || 0, 1)} ms avg</span>
          </div>
          <div class="sparkline-card__chart">${sparklineSvg(latencySeries, { stroke: 'var(--accent-2)', fill: 'rgba(56, 189, 248, 0.14)' })}</div>
        </section>
        <section class="sparkline-card">
          <div class="sparkline-card__title">
            <strong>Answer correctness</strong>
            <span class="chip">${evals.length ? formatNumber((evals.reduce((sum, entry) => sum + Number(entry.answer_correctness || 0), 0) / evals.length) || 0, 2) : '0.00'}</span>
          </div>
          <div class="sparkline-card__chart">${sparklineSvg(correctnessSeries, { stroke: 'var(--success)', fill: 'rgba(52, 211, 153, 0.12)' })}</div>
        </section>
      </div>
      <section class="card">
        <div class="card__header">
          <h2>Recent evaluations</h2>
          <span class="chip">${formatNumber(evals.length)} samples</span>
        </div>
        <div class="card__body">
          ${evals.length ? renderEvaluationTable(evals) : renderEmptyBlock('No evaluations yet', 'Ask a question to populate answer quality, faithfulness, and latency metrics.')}
        </div>
      </section>
      <section class="card">
        <div class="card__header">
          <h2>Raw Prometheus snapshot</h2>
          <button class="button button--small button--ghost" type="button" data-action="copy-metrics">Copy metrics</button>
        </div>
        <div class="card__body">
          <pre class="message__body mono" style="white-space: pre-wrap; margin: 0;">${escapeHtml(state.overview?.metrics || '')}</pre>
        </div>
      </section>
    </div>
  `;
}

function renderEvaluationTable(evaluations) {
  return `
    <div class="list">
      ${evaluations
        .map(
          (evaluation) => `
            <button class="list__item clickable" type="button" data-evaluation-json="${escapeHtml(JSON.stringify(evaluation))}">
              <span class="avatar">E</span>
              <span style="min-width:0;">
                <strong>${escapeHtml(truncate(evaluation.query || 'Evaluation', 80))}</strong>
                <small>${formatRelativeTime(evaluation.created_at)} · correctness ${formatNumber(evaluation.answer_correctness || 0, 2)} · latency ${formatNumber(evaluation.latency_ms || 0, 1)} ms</small>
              </span>
            </button>
          `,
        )
        .join('')}
    </div>
  `;
}

function renderSettingsSection() {
  const featureFlags = state.overview?.feature_flags || {};
  return `
    <div class="section-grid dual">
      <section class="card">
        <div class="card__header">
          <h2>Connection and theme</h2>
          <span class="chip chip--accent">${state.authState === 'ok' ? 'Authenticated' : 'Needs key'}</span>
        </div>
        <div class="card__body">
          <form class="form-grid" id="settings-form">
            <div class="field">
              <label for="api-key-input">API key</label>
              <input id="api-key-input" class="input" type="password" value="${escapeHtml(state.apiKey)}" placeholder="Bearer token used by the FastAPI middleware" />
            </div>
            <div class="field">
              <label for="theme-select">Theme</label>
              <select id="theme-select" class="input">
                <option value="system" ${state.themeMode === 'system' ? 'selected' : ''}>System</option>
                <option value="dark" ${state.themeMode === 'dark' ? 'selected' : ''}>Dark</option>
                <option value="light" ${state.themeMode === 'light' ? 'selected' : ''}>Light</option>
              </select>
            </div>
            <div class="toolbar">
              <div class="toolbar__group">
                <button class="button button--small button--primary" type="submit">Save settings</button>
                <button class="button button--small button--ghost" type="button" data-action="reload-workspace">Reload</button>
              </div>
              <span class="chip">This workspace uses same-origin API calls</span>
            </div>
          </form>
        </div>
      </section>
      <section class="card">
        <div class="card__header">
          <h2>Feature flags</h2>
          <span class="chip">Backend settings</span>
        </div>
        <div class="card__body">
          <div class="list">
            ${Object.entries(featureFlags)
              .map(
                ([key, value]) => `
                  <div class="toggle-row">
                    <div>
                      <strong>${escapeHtml(key.replace(/_/g, ' '))}</strong>
                      <small>${value ? 'Enabled' : 'Disabled'}</small>
                    </div>
                    <div class="switch ${value ? 'is-on' : ''}" aria-hidden="true"></div>
                  </div>
                `,
              )
              .join('')}
          </div>
        </div>
      </section>
      <section class="card">
        <div class="card__header">
          <h2>Session utilities</h2>
        </div>
        <div class="card__body">
          <div class="list">
            <div class="list__item">
              <span class="avatar">S</span>
              <span style="min-width:0;">
                <strong>Shortcut hints</strong>
                <small>⌘K for search, ⌘↵ to send, / for commands, Esc to exit full-screen fields.</small>
              </span>
            </div>
            <div class="list__item">
              <span class="avatar">P</span>
              <span style="min-width:0;">
                <strong>Persistence</strong>
                <small>Theme, pinned chats, active conversation, and draft board are stored locally.</small>
              </span>
            </div>
            <div class="list__item">
              <span class="avatar">R</span>
              <span style="min-width:0;">
                <strong>Response mode</strong>
                <small>Streaming NDJSON is used when available; the interface falls back to regular JSON.</small>
              </span>
            </div>
          </div>
        </div>
      </section>
      <section class="card">
        <div class="card__header">
          <h2>Environment</h2>
        </div>
        <div class="card__body">
          <div class="stat-grid">
            <div class="stat">
              <div class="stat__label">Workspace</div>
              <div class="stat__value">${escapeHtml(state.overview?.version || 'dev')}</div>
            </div>
            <div class="stat">
              <div class="stat__label">Conversation ID</div>
              <div class="stat__value">${escapeHtml(shortId(state.activeConversationId || 'none'))}</div>
            </div>
          </div>
        </div>
      </section>
    </div>
  `;
}

function renderCopilotContext() {
  const chips = [];
  chips.push(`<span class="chip chip--accent">${escapeHtml(state.activeSection || 'dashboard')}</span>`);
  if (state.activeConversationId) {
    chips.push(`<span class="chip">Conversation ${escapeHtml(shortId(state.activeConversationId))}</span>`);
  }
  if (state.selectedDocument) {
    chips.push(`<span class="chip">Document ${escapeHtml(truncate(state.selectedDocument.filename || 'document', 24))}</span>`);
  }
  if (state.selectedSource?.metadata?.source_filename) {
    chips.push(`<span class="chip">Source ${escapeHtml(truncate(state.selectedSource.metadata.source_filename, 24))}</span>`);
  }
  return chips.join('');
}

function renderCopilotPanel() {
  const messages = getVisibleMessages();
  const placeholder = PLACEHOLDERS[currentPlaceholderIndex % PLACEHOLDERS.length];
  const emptyState = renderCopilotEmptyState();
  return `
    <div class="messages copilot__messages" id="copilot-messages">
      ${state.loading || (!messages.length && !state.streaming) ? emptyState : messages.map((message, index) => renderCopilotMessage(message, index)).join('')}
      ${state.streaming ? renderStreamingSkeleton() : ''}
    </div>
    <form class="composer" id="copilot-form">
      <div class="composer__box">
        <div class="composer__input">
          <textarea id="copilot-prompt" class="input" rows="4" placeholder="${escapeHtml(placeholder)}">${escapeHtml(state.draft)}</textarea>
          <div class="composer__toolbar">
            <div class="composer__chips">
              ${COMMANDS.map((command) => `<button class="chip clickable" type="button" data-command="${escapeHtml(command.prompt)}">${escapeHtml(command.label)}</button>`).join('')}
            </div>
            <div class="composer__actions">
              <button class="button button--small button--ghost" type="button" data-action="voice-input">Voice</button>
              <button class="button button--small button--ghost" type="button" data-action="attach-file">File</button>
              <button class="button button--small button--ghost" type="button" data-action="attach-folder">Folder</button>
              <button class="button button--small button--primary" type="submit" ${state.streaming ? 'disabled' : ''}>Send</button>
            </div>
          </div>
        </div>
      </div>
    </form>
  `;
}

function renderCopilotEmptyState() {
  const suggestions = buildSuggestions();
  return `
    <div class="empty-state">
      <h3>${escapeHtml(suggestions.title)}</h3>
      <p>${escapeHtml(suggestions.subtitle)}</p>
      <div class="empty-state__actions">
        ${suggestions.actions
          .map((action) => `<button class="button button--small button--ghost" type="button" data-prompt="${escapeHtml(action)}">${escapeHtml(action)}</button>`)
          .join('')}
      </div>
    </div>
  `;
}

function renderStreamingSkeleton() {
  return `
    <article class="message message--assistant">
      <div class="message__meta">
        <strong>Assistant</strong>
        <span class="chip chip--accent">Streaming</span>
      </div>
      <div class="assistant-skeleton">
        <div class="skeleton-line" style="width: 88%;"></div>
        <div class="skeleton-line" style="width: 72%;"></div>
        <div class="skeleton-line" style="width: 58%;"></div>
      </div>
    </article>
  `;
}

function renderCopilotMessage(message, index) {
  const role = message.role || 'assistant';
  const isAssistant = role === 'assistant';
  const sources = message.sources || [];
  const citations = message.citations || [];
  const actions = isAssistant
    ? `
      <div class="message__actions">
        <button class="button button--small button--ghost" type="button" data-action="copy-message" data-content="${escapeHtml(message.content || '')}">Copy</button>
        <button class="button button--small button--ghost" type="button" data-action="insert-message" data-content="${escapeHtml(message.content || '')}">Insert</button>
        <button class="button button--small button--ghost" type="button" data-action="download-message" data-content="${escapeHtml(message.content || '')}" data-role="Assistant">Download</button>
        ${message.prompt ? `<button class="button button--small button--ghost" type="button" data-action="explain-message" data-prompt="${escapeHtml(message.prompt)}">Explain</button>` : ''}
      </div>
    `
    : '';

  const sourcesHtml = sources.length
    ? `
      <div class="message__sources">
        ${sources
          .map(
            (source, sourceIndex) => `
              <button class="source-card clickable ${state.selectedSourceIndex === sourceIndex && state.selectedSource?.content === (source.content || source.page_content || '') ? 'is-selected' : ''}" type="button" data-source-index="${sourceIndex}" data-source-json="${escapeHtml(JSON.stringify(source))}">
                <div class="source-card__top">
                  <span>${escapeHtml(source.metadata?.source_filename || source.metadata?.filename || `Source ${sourceIndex + 1}`)}</span>
                  <span>${source.score != null ? `score ${formatNumber(source.score, 3)}` : 'source'}</span>
                </div>
                <div class="source-card__excerpt">${escapeHtml(truncate(source.content || source.page_content || '', 260))}</div>
              </button>
            `,
          )
          .join('')}
      </div>
    `
    : '';

  const citationHtml = citations.length
    ? `<div class="copilot__context">${citations.map((citation) => `<span class="chip chip--accent">${escapeHtml(citation.citation_id || 'citation')}</span>`).join('')}</div>`
    : '';

  return `
    <article class="message ${isAssistant ? 'message--assistant' : ''}">
      <div class="message__meta">
        <strong>${escapeHtml(role.charAt(0).toUpperCase() + role.slice(1))}</strong>
        <div class="toolbar__group">
          ${message.created_at ? `<span class="chip">${formatDateTime(message.created_at)}</span>` : ''}
          ${message.tokens_used != null ? `<span class="chip">${formatNumber(message.tokens_used)} tokens</span>` : ''}
        </div>
      </div>
      <div class="message__body">${markdownToHtml(message.content || '')}</div>
      ${citationHtml}
      ${sourcesHtml}
      ${actions}
    </article>
  `;
}

function getVisibleMessages() {
  if (state.streaming && state.conversationDetail?.messages?.length) {
    return state.conversationDetail.messages;
  }
  if (state.conversationDetail?.messages?.length) {
    return state.conversationDetail.messages;
  }
  return [];
}

function buildSuggestions() {
  const docs = state.overview?.recent_documents || [];
  const failed = docs.filter((doc) => doc.status === 'failed');
  if (!docs.length) {
    return {
      title: 'No chat history yet',
      subtitle: 'Ask a question, upload a document, or run a semantic search to seed the workspace.',
      actions: QUICK_PROMPTS.slice(0, 3),
    };
  }
  const actions = QUICK_PROMPTS.slice();
  if (failed.length) {
    actions.unshift('Investigate failed ingests and fix metadata issues.');
  }
  if (docs.some((doc) => doc.has_ocr)) {
    actions.unshift('Summarize the latest OCR-heavy document.');
  }
  return {
    title: 'Suggested next moves',
    subtitle: 'The copilot has enough context to help you summarize, search, compare, or export.',
    actions: actions.slice(0, 4),
  };
}

function filterConversations(conversations) {
  const needle = state.search.trim().toLowerCase();
  if (!needle) {
    return conversations;
  }
  return conversations.filter((conversation) => {
    const haystack = [
      conversation.title,
      conversation.conversation_id,
      conversation.message_count,
      conversation.updated_at,
    ]
      .map((item) => String(item ?? '').toLowerCase())
      .join(' ');
    return haystack.includes(needle);
  });
}

function makeGreeting() {
  const hour = new Date().getHours();
  const title = hour < 12 ? 'Good morning.' : hour < 18 ? 'Good afternoon.' : 'Good evening.';
  const docs = state.overview?.counts?.documents || 0;
  const conversations = state.overview?.counts?.conversations || 0;
  const evals = state.overview?.counts?.evaluations || 0;
  return {
    title,
    subtitle: docs
      ? `This workspace currently tracks ${formatNumber(docs)} documents, ${formatNumber(conversations)} conversations, and ${formatNumber(evals)} evaluations. The interface is tuned for fast inspection, grounded answers, and production telemetry.`
      : 'Start by uploading a document or asking a question. The interface is ready for streaming answers, citations, and document intelligence.',
  };
}

function showToast(title, detail = '', tone = 'info') {
  const toast = document.createElement('div');
  toast.className = 'toast';
  if (tone === 'danger') {
    toast.style.borderColor = 'rgba(251, 113, 133, 0.3)';
  } else if (tone === 'success') {
    toast.style.borderColor = 'rgba(52, 211, 153, 0.28)';
  }
  toast.innerHTML = `
    <strong>${escapeHtml(title)}</strong>
    ${detail ? `<p>${escapeHtml(detail)}</p>` : ''}
  `;
  els.toastHost.appendChild(toast);
  setTimeout(() => toast.remove(), 3200);
}

async function refreshWorkspace() {
  if (!state.apiKey) {
    state.authState = 'missing';
    state.loading = false;
    scheduleRender();
    return;
  }
  api.setApiKey(state.apiKey);
  try {
    const [overview, conversationsResponse] = await Promise.all([
      api.overview(),
      api.conversations(1, 100),
    ]);
    state.overview = overview;
    state.parsedMetrics = parsePrometheusMetrics(overview.metrics || '');
    state.metricsText = overview.metrics || '';
    state.conversations = conversationsResponse.conversations || [];
    state.authState = 'ok';
    state.loading = false;
    if (!state.activeConversationId && state.conversations.length) {
      state.activeConversationId = state.conversations[0].conversation_id;
      localStorage.setItem('rag_active_conversation', state.activeConversationId);
      await loadConversation(state.activeConversationId, { silent: true });
    } else if (state.activeConversationId) {
      await loadConversation(state.activeConversationId, { silent: true });
    }
    if (!state.selectedDocument && overview.recent_documents?.length) {
      state.selectedDocument = overview.recent_documents[0];
    }
    scheduleRender();
  } catch (error) {
    state.loading = false;
    if (error instanceof ApiError && (error.status === 401 || error.status === 403)) {
      state.authState = 'invalid';
      showToast('Authentication failed', 'Paste a valid API key in Settings.', 'danger');
    } else {
      state.authState = 'error';
      showToast('Workspace sync failed', error.message || 'Unable to sync the dashboard.', 'danger');
    }
    scheduleRender();
  }
}

async function loadConversation(conversationId, options = {}) {
  if (!conversationId) {
    state.conversationDetail = null;
    state.selectedSource = null;
    state.selectedSourceIndex = -1;
    if (!options.silent) scheduleRender();
    return;
  }
  try {
    state.loadingConversation = true;
    if (!options.silent) scheduleRender();
    const detail = await api.conversationDetail(conversationId);
    state.conversationDetail = detail;
    state.activeConversationId = detail.conversation_id;
    localStorage.setItem('rag_active_conversation', detail.conversation_id);
    state.selectedSource = null;
    state.selectedSourceIndex = -1;
    state.loadingConversation = false;
    if (!options.silent) scheduleRender();
  } catch (error) {
    state.loadingConversation = false;
    if (!options.silent) {
      showToast('Conversation unavailable', error.message || 'Unable to load conversation details.', 'danger');
      scheduleRender();
    }
  }
}

async function loadAndRenderEvaluationSamples() {
  try {
    const response = await api.evaluations(12);
    state.overview = {
      ...state.overview,
      recent_evaluations: response.evaluations || [],
    };
    scheduleRender();
  } catch {
    return;
  }
}

function bindStaticEvents() {
  document.addEventListener('click', handleClick);
  document.addEventListener('submit', handleSubmit);
  document.addEventListener('input', handleInput);
  document.addEventListener('change', handleChange);
  document.addEventListener('keydown', handleKeydown);
  window.addEventListener('hashchange', handleHashChange);
  window.addEventListener('paste', handlePaste);
  window.addEventListener('dragover', handleDragOver);
  window.addEventListener('drop', handleDrop);
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    if (state.themeMode === 'system') {
      applyLayoutPrefs();
    }
  });
}

function handleHashChange() {
  const section = location.hash.replace('#', '') || 'dashboard';
  state.activeSection = section;
  scheduleRender();
}

function handleClick(event) {
  const target = event.target.closest('[data-action], [data-section], [data-prompt], [data-command], [data-conversation-id], [data-document-id], [data-source-json], [data-source-index], [data-evaluation-json]');
  if (!target) {
    return;
  }

  const action = target.getAttribute('data-action');
  const section = target.getAttribute('data-section');
  const prompt = target.getAttribute('data-prompt');
  const command = target.getAttribute('data-command');
  const conversationId = target.getAttribute('data-conversation-id');
  const documentId = target.getAttribute('data-document-id');
  const sourceJson = target.getAttribute('data-source-json');
  const sourceIndex = target.getAttribute('data-source-index');
  const evaluationJson = target.getAttribute('data-evaluation-json');

  if (section) {
    event.preventDefault();
    setSection(section);
    return;
  }

  if (prompt) {
    event.preventDefault();
    setDraft(prompt);
    return;
  }

  if (command) {
    event.preventDefault();
    setDraft(command);
    focusComposer();
    return;
  }

  if (conversationId) {
    event.preventDefault();
    openConversation(conversationId);
    return;
  }

  if (documentId) {
    event.preventDefault();
    selectDocument(documentId);
    return;
  }

  if (sourceJson) {
    event.preventDefault();
    selectSource(JSON.parse(sourceJson), Number(sourceIndex || 0));
    return;
  }

  if (evaluationJson) {
    event.preventDefault();
    state.selectedEvaluation = JSON.parse(evaluationJson);
    state.activeSection = 'metrics';
    scheduleRender();
    showToast('Evaluation selected', 'Scroll the metrics panel for the full telemetry snapshot.');
    return;
  }

  switch (action) {
    case 'refresh-all':
      event.preventDefault();
      refreshWorkspace();
      loadAndRenderEvaluationSamples();
      return;
    case 'create-conversation':
      event.preventDefault();
      createConversation();
      return;
    case 'upload-file':
      event.preventDefault();
      triggerUpload(false);
      return;
    case 'upload-folder':
      event.preventDefault();
      triggerUpload(true);
      return;
    case 'paste-upload':
      event.preventDefault();
      showToast('Paste ready', 'Paste an image or file to upload it directly into the workspace.');
      return;
    case 'theme-cycle':
      event.preventDefault();
      cycleTheme();
      return;
    case 'open-settings':
      event.preventDefault();
      setSection('settings');
      return;
    case 'toggle-copilot':
      event.preventDefault();
      toggleCopilot();
      return;
    case 'copy-message':
      event.preventDefault();
      copyText(target.getAttribute('data-content') || '').then(() => showToast('Copied', 'Message text copied to clipboard.', 'success'));
      return;
    case 'insert-message':
      event.preventDefault();
      insertDraft(target.getAttribute('data-content') || '');
      return;
    case 'download-message':
      event.preventDefault();
      downloadText(`${safeFileName(target.getAttribute('data-role') || 'assistant')}.md`, target.getAttribute('data-content') || '');
      return;
    case 'retry-message':
      event.preventDefault();
      setDraft(target.getAttribute('data-prompt') || '');
      submitPrompt(target.getAttribute('data-prompt') || '', true);
      return;
    case 'explain-message':
      event.preventDefault();
      submitPrompt(`Explain the following in plain English with evidence: ${target.getAttribute('data-prompt') || ''}`, true);
      return;
    case 'rename-conversation':
      event.preventDefault();
      renameConversation();
      return;
    case 'pin-conversation':
      event.preventDefault();
      togglePinnedConversation();
      return;
    case 'export-conversation':
      event.preventDefault();
      exportConversation();
      return;
    case 'delete-conversation':
      event.preventDefault();
      deleteConversation();
      return;
    case 'copy-metrics':
      event.preventDefault();
      copyText(state.metricsText || '').then(() => showToast('Copied', 'Prometheus snapshot copied to clipboard.', 'success'));
      return;
    case 'clear-draft':
      event.preventDefault();
      setDraftBoard('');
      return;
    case 'copy-draft':
      event.preventDefault();
      copyText(state.draftBoard || '').then(() => showToast('Copied', 'Draft copied to clipboard.', 'success'));
      return;
    case 'download-draft-md':
      event.preventDefault();
      downloadText('draft.md', state.draftBoard || '');
      return;
    case 'download-draft-json':
      event.preventDefault();
      downloadText('draft.json', JSON.stringify({ draft: state.draftBoard || '' }, null, 2), 'application/json');
      return;
    case 'reload-workspace':
      event.preventDefault();
      refreshWorkspace();
      return;
    case 'voice-input':
      event.preventDefault();
      startVoiceInput();
      return;
    case 'attach-file':
      event.preventDefault();
      triggerUpload(false);
      return;
    case 'attach-folder':
      event.preventDefault();
      triggerUpload(true);
      return;
    case 'submit-prompt':
      event.preventDefault();
      submitCurrentPrompt();
      return;
    case 'toggle-theme-mode':
      event.preventDefault();
      cycleTheme();
      return;
    case 'toggle-side':
      event.preventDefault();
      toggleCopilot();
      return;
    default:
      return;
  }
}

function handleSubmit(event) {
  const form = event.target.closest('#copilot-form, #settings-form');
  if (!form) {
    return;
  }
  event.preventDefault();
  if (form.id === 'copilot-form') {
    submitCurrentPrompt();
  }
  if (form.id === 'settings-form') {
    saveSettings(form);
  }
}

function handleInput(event) {
  const target = event.target;
  if (!(target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement)) {
    return;
  }

  if (target.id === 'global-search' || target.id === 'conversation-search') {
    state.search = target.value;
    scheduleRender();
    return;
  }

  if (target.id === 'copilot-prompt') {
    setDraft(target.value, { silent: true });
    return;
  }

  if (target.id === 'draft-board') {
    setDraftBoard(target.value, { silent: true });
    return;
  }
}

function handleChange(event) {
  const target = event.target;
  if (!(target instanceof HTMLInputElement || target instanceof HTMLSelectElement)) {
    return;
  }
  if (target.id === 'theme-select') {
    setThemeMode(target.value);
    scheduleRender();
  }
}

function handleKeydown(event) {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault();
    document.getElementById('global-search')?.focus();
    return;
  }
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
    const prompt = document.getElementById('copilot-prompt');
    if (prompt && document.activeElement === prompt) {
      event.preventDefault();
      submitCurrentPrompt();
    }
  }
  if (event.key === '/' && document.activeElement === document.body) {
    const prompt = document.getElementById('copilot-prompt');
    if (prompt) {
      event.preventDefault();
      prompt.focus();
      setDraft('/');
    }
  }
}

function handlePaste(event) {
  const items = Array.from(event.clipboardData?.files || []);
  if (items.length) {
    event.preventDefault();
    processUpload(items);
  }
}

function handleDragOver(event) {
  if (event.dataTransfer?.types?.includes('Files')) {
    event.preventDefault();
    setUploadDrag(true);
  }
}

function handleDrop(event) {
  const files = Array.from(event.dataTransfer?.files || []);
  if (files.length) {
    event.preventDefault();
    setUploadDrag(false);
    processUpload(files);
  }
}

function setSection(section) {
  state.activeSection = section;
  location.hash = section === 'dashboard' ? '' : section;
  scheduleRender();
}

function setDraft(value, options = {}) {
  state.draft = value;
  if (!options.silent) {
    scheduleRender();
  }
}

function setDraftBoard(value, options = {}) {
  state.draftBoard = value;
  localStorage.setItem('rag_draft_board', state.draftBoard);
  if (!options.silent) {
    scheduleRender();
  }
}

function cycleTheme() {
  const order = ['system', 'dark', 'light'];
  const index = Math.max(order.indexOf(state.themeMode), 0);
  const next = order[(index + 1) % order.length];
  setThemeMode(next);
  showToast('Theme updated', `Theme mode changed to ${next}.`, 'success');
  scheduleRender();
}

function toggleCopilot() {
  state.rightCollapsed = !state.rightCollapsed;
  localStorage.setItem('rag_right_collapsed', state.rightCollapsed ? '1' : '0');
  scheduleRender();
}

function setUploadDrag(active) {
  state.upload.active = active;
  scheduleRender();
}

function setLayoutWidth(width) {
  state.rightWidth = Math.max(320, Math.min(620, Math.round(width)));
  document.documentElement.style.setProperty('--right-panel-width', `${state.rightWidth}px`);
  localStorage.setItem('rag_right_width', String(state.rightWidth));
}

function triggerUpload(folder = false) {
  const input = document.createElement('input');
  input.type = 'file';
  input.multiple = true;
  if (folder) {
    input.setAttribute('webkitdirectory', '');
    input.setAttribute('directory', '');
  }
  input.addEventListener('change', () => {
    const files = Array.from(input.files || []);
    if (files.length) {
      processUpload(files);
    }
  });
  input.click();
}

function startVoiceInput() {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) {
    showToast('Voice input unavailable', 'This browser does not support the SpeechRecognition API.', 'danger');
    return;
  }
  if (activeVoiceRecognition) {
    activeVoiceRecognition.stop();
    activeVoiceRecognition = null;
    showToast('Voice input stopped', 'Speech recognition was stopped.');
    return;
  }
  const recognition = new Recognition();
  recognition.lang = 'en-US';
  recognition.interimResults = true;
  recognition.continuous = false;
  recognition.onresult = (event) => {
    const transcript = Array.from(event.results)
      .map((result) => result[0].transcript)
      .join(' ')
      .trim();
    const prompt = document.getElementById('copilot-prompt');
    if (prompt) {
      prompt.value = transcript;
      setDraft(transcript, { silent: true });
    }
  };
  recognition.onerror = () => {
    showToast('Voice input failed', 'Speech recognition could not be started.', 'danger');
    activeVoiceRecognition = null;
  };
  recognition.onend = () => {
    activeVoiceRecognition = null;
  };
  recognition.start();
  activeVoiceRecognition = recognition;
  showToast('Voice input active', 'Speak now and the transcript will fill the prompt.');
}

function focusComposer() {
  const prompt = document.getElementById('copilot-prompt');
  prompt?.focus();
}

function insertDraft(content) {
  const next = [state.draftBoard.trim(), content.trim()].filter(Boolean).join('\n\n---\n\n');
  setDraftBoard(next);
  setSection('dashboard');
  showToast('Inserted', 'Answer text added to the working draft.', 'success');
}

function safeFileName(value) {
  return String(value || 'download').replace(/[^\w.-]+/g, '_');
}

function exportConversation() {
  const detail = state.conversationDetail;
  if (!detail) {
    return;
  }
  const markdown = [
    `# ${detail.title || 'Conversation'}`,
    '',
    `- Conversation ID: ${detail.conversation_id}`,
    `- Updated: ${formatDateTime(detail.updated_at)}`,
    '',
    ...(detail.messages || []).map((message) => `## ${message.role}\n\n${message.content}\n`),
  ].join('\n');
  downloadText(`${safeFileName(detail.title || 'conversation')}.md`, markdown);
}

function renameConversation() {
  const detail = state.conversationDetail;
  if (!detail) {
    return;
  }
  const next = window.prompt('Rename conversation', detail.title || 'New conversation');
  if (!next) {
    return;
  }
  api.updateConversation(detail.conversation_id, next.trim())
    .then((updated) => {
      showToast('Conversation renamed', updated.title, 'success');
      refreshWorkspace();
    })
    .catch((error) => showToast('Rename failed', error.message || 'Unable to rename conversation.', 'danger'));
}

function togglePinnedConversation() {
  const id = state.conversationDetail?.conversation_id;
  if (!id) {
    return;
  }
  const pinned = new Set(state.pinnedConversationIds);
  if (pinned.has(id)) {
    pinned.delete(id);
    showToast('Conversation unpinned', 'Removed from quick access.');
  } else {
    pinned.add(id);
    showToast('Conversation pinned', 'Pinned to the sidebar.', 'success');
  }
  state.pinnedConversationIds = Array.from(pinned);
  localStorage.setItem('rag_pinned_conversations', JSON.stringify(state.pinnedConversationIds));
  scheduleRender();
}

function deleteConversation() {
  const id = state.conversationDetail?.conversation_id;
  if (!id) {
    return;
  }
  if (!window.confirm('Delete this conversation and its stored messages?')) {
    return;
  }
  api.deleteConversation(id)
    .then(() => {
      showToast('Conversation deleted', 'History removed from the workspace.', 'success');
      state.conversationDetail = null;
      state.activeConversationId = '';
      localStorage.removeItem('rag_active_conversation');
      refreshWorkspace();
    })
    .catch((error) => showToast('Delete failed', error.message || 'Unable to delete conversation.', 'danger'));
}

function selectDocument(documentId) {
  const docs = state.overview?.recent_documents || [];
  const selected = docs.find((document) => (document.document_id || document._id || document.filename) === documentId);
  if (!selected) {
    return;
  }
  state.selectedDocument = selected;
  state.activeSection = 'documents';
  scheduleRender();
}

function selectSource(source, sourceIndex = 0) {
  state.selectedSource = source;
  state.selectedSourceIndex = sourceIndex;
  if (source?.metadata) {
    const docId = source.metadata.document_id || source.metadata._id;
    const docs = state.overview?.recent_documents || [];
    const doc = docs.find((entry) => (entry.document_id || entry._id) === docId);
    if (doc) {
      state.selectedDocument = doc;
    }
  }
  state.activeSection = 'documents';
  scheduleRender();
}

async function openConversation(conversationId) {
  state.activeConversationId = conversationId;
  localStorage.setItem('rag_active_conversation', conversationId);
  await loadConversation(conversationId);
}

function saveSettings(form) {
  const apiKeyInput = form.querySelector('#api-key-input');
  const themeSelect = form.querySelector('#theme-select');
  if (apiKeyInput instanceof HTMLInputElement) {
    state.apiKey = apiKeyInput.value.trim();
    if (state.apiKey) {
      localStorage.setItem('rag_api_key', state.apiKey);
    } else {
      localStorage.removeItem('rag_api_key');
    }
  }
  if (themeSelect instanceof HTMLSelectElement) {
    setThemeMode(themeSelect.value);
  }
  showToast('Settings saved', 'Theme and API key preferences were updated.', 'success');
  refreshWorkspace();
}

function submitCurrentPrompt() {
  const prompt = document.getElementById('copilot-prompt');
  const value = prompt instanceof HTMLTextAreaElement ? prompt.value.trim() : state.draft.trim();
  if (!value) {
    showToast('Prompt required', 'Type a question or choose a quick action.', 'danger');
    return;
  }
  setDraft(value, { silent: true });
  submitPrompt(value, false);
}

function appendLocalMessage(role, content, extra = {}) {
  const messages = state.conversationDetail?.messages ? [...state.conversationDetail.messages] : [];
  messages.push({
    role,
    content,
    created_at: new Date().toISOString(),
    sources: extra.sources || [],
    tokens_used: extra.tokens_used || 0,
  });
  state.conversationDetail = {
    ...(state.conversationDetail || {
      conversation_id: state.activeConversationId || '',
      title: 'Conversation',
      message_count: 0,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }),
    messages,
    message_count: messages.length,
    updated_at: new Date().toISOString(),
  };
}

async function submitPrompt(prompt, isRegenerated = false) {
  if (state.streaming) {
    showToast('Still streaming', 'Wait for the current answer to finish.');
    return;
  }
  if (!state.apiKey) {
    showToast('API key required', 'Add a valid bearer token in Settings first.', 'danger');
    setSection('settings');
    return;
  }
  api.setApiKey(state.apiKey);

  const trimmed = prompt.trim();
  if (!trimmed) {
    return;
  }

  state.streaming = true;
  state.streamError = '';
  state.streamMetadata = null;
  state.selectedSource = null;
  state.selectedSourceIndex = -1;
  setDraft('');
  setSection('conversations');
  appendLocalMessage('user', trimmed);
  state.conversationDetail.messages.push({
    role: 'assistant',
    content: '',
    created_at: new Date().toISOString(),
    sources: [],
    tokens_used: 0,
    streaming: true,
    prompt: trimmed,
  });
  scheduleRender();

  const payload = {
    query: trimmed,
    conversation_id: state.activeConversationId || undefined,
    top_k: 6,
  };

  try {
    let assistantIndex = state.conversationDetail.messages.length - 1;
    let assistantText = '';
    let citations = [];
    let finalVerification = null;
    let finalConfidence = null;
    let finalConversationId = state.activeConversationId || '';
    let streamFallback = false;

    try {
      for await (const event of api.streamQuery(payload)) {
        if (event.type === 'metadata') {
          finalConversationId = event.conversation_id || finalConversationId;
          citations = event.citations || [];
          state.streamMetadata = event;
          if (finalConversationId && finalConversationId !== state.activeConversationId) {
            state.activeConversationId = finalConversationId;
            localStorage.setItem('rag_active_conversation', finalConversationId);
          }
          scheduleRender();
          continue;
        }
        if (event.type === 'token') {
          assistantText += event.content || '';
          state.conversationDetail.messages[assistantIndex].content = assistantText;
          state.conversationDetail.messages[assistantIndex].citations = citations;
          scheduleRender();
          continue;
        }
        if (event.type === 'done') {
          finalVerification = event.verification || null;
          finalConfidence = event.confidence_score ?? null;
          if (event.conversation_id) {
            finalConversationId = event.conversation_id;
            state.activeConversationId = finalConversationId;
            localStorage.setItem('rag_active_conversation', finalConversationId);
          }
        }
      }
    } catch (error) {
      streamFallback = true;
      if (!(error instanceof ApiError)) {
        throw error;
      }
      const fallback = await api.query(payload);
      finalConversationId = fallback.conversation_id || finalConversationId;
      assistantText = fallback.answer || '';
      citations = fallback.citations || [];
      finalVerification = fallback.verification || null;
      finalConfidence = fallback.confidence_score ?? null;
      state.conversationDetail.messages[assistantIndex] = {
        role: 'assistant',
        content: assistantText,
        created_at: new Date().toISOString(),
        sources: fallback.sources || [],
        tokens_used: fallback.tokens_used || 0,
        citations,
        verification: finalVerification,
        confidence_score: finalConfidence,
        prompt: trimmed,
      };
      state.activeConversationId = finalConversationId;
      localStorage.setItem('rag_active_conversation', finalConversationId);
      scheduleRender();
    }

    if (!streamFallback) {
      state.conversationDetail.messages[assistantIndex] = {
        role: 'assistant',
        content: assistantText,
        created_at: new Date().toISOString(),
        sources: state.conversationDetail.messages[assistantIndex].sources || [],
        tokens_used: 0,
        citations,
        verification: finalVerification,
        confidence_score: finalConfidence,
        prompt: trimmed,
      };
    }

    await refreshWorkspace();
    state.streaming = false;
    scheduleRender();
    showToast('Answer ready', isRegenerated ? 'Regenerated response delivered.' : 'Streaming answer completed.', 'success');
  } catch (error) {
    state.streaming = false;
    state.streamError = error.message || 'Request failed';
    state.conversationDetail.messages = state.conversationDetail.messages.filter((message) => !message.streaming);
    scheduleRender();
    showToast('Query failed', error.message || 'The RAG request could not be completed.', 'danger');
  }
}

function setSelectedMessageSource(sourceJson) {
  try {
    const source = typeof sourceJson === 'string' ? JSON.parse(sourceJson) : sourceJson;
    if (source) {
      state.selectedSource = source;
      state.activeSection = 'documents';
      scheduleRender();
    }
  } catch {
    return;
  }
}

async function processUpload(files) {
  if (!state.apiKey) {
    showToast('API key required', 'You need an API key before uploading documents.', 'danger');
    setSection('settings');
    return;
  }
  if (!files.length) {
    return;
  }
  api.setApiKey(state.apiKey);

  for (const file of files) {
    state.upload = {
      active: true,
      filename: file.name,
      progress: 0,
      stage: 'Uploading',
      status: 'processing',
      error: '',
    };
    scheduleRender();

    let stageIndex = 0;
    clearInterval(uploadStageTimer);
    uploadStageTimer = window.setInterval(() => {
      if (!state.upload.active) {
        clearInterval(uploadStageTimer);
        return;
      }
      if (state.upload.progress >= 100) {
        stageIndex = Math.min(stageIndex + 1, uploadMilestones.length - 1);
        state.upload.stage = uploadMilestones[stageIndex];
        scheduleRender();
      }
    }, 900);

    try {
      const result = await api.uploadFile(file, (progress) => {
        state.upload.progress = Math.max(state.upload.progress, progress);
        if (progress >= 100) {
          state.upload.stage = 'Processing';
        }
        scheduleRender();
      });

      clearInterval(uploadStageTimer);
      state.upload = {
        active: false,
        filename: file.name,
        progress: 100,
        stage: 'Complete',
        status: 'success',
        error: '',
      };
      scheduleRender();
      showToast('Upload complete', `${file.name} ingested successfully.`, 'success');
      if (result?.document_id) {
        state.activeSection = 'documents';
      }
      await refreshWorkspace();
    } catch (error) {
      clearInterval(uploadStageTimer);
      state.upload = {
        active: false,
        filename: file.name,
        progress: 0,
        stage: 'Failed',
        status: 'error',
        error: error.message || 'Upload failed',
      };
      scheduleRender();
      showToast('Upload failed', error.message || 'Document ingestion failed.', 'danger');
      break;
    }
  }
}

function renderConversationMiniList(conversations, compact = false) {
  return `
    <div class="list ${compact ? 'list--dense' : ''}">
      ${conversations
        .map(
          (conversation) => `
            <button class="list__item clickable ${state.activeConversationId === conversation.conversation_id ? 'list__item--active' : ''}" type="button" data-conversation-id="${escapeHtml(conversation.conversation_id)}">
              <span class="avatar">${escapeHtml((conversation.title || 'C').charAt(0))}</span>
              <span style="min-width:0;">
                <strong>${escapeHtml(conversation.title || 'Untitled')}</strong>
                <small>${formatNumber(conversation.message_count || 0)} messages · ${formatRelativeTime(conversation.updated_at)}</small>
              </span>
            </button>
          `,
        )
        .join('')}
    </div>
  `;
}

function setActiveConversationFromDetail() {
  if (state.conversationDetail?.conversation_id) {
    state.activeConversationId = state.conversationDetail.conversation_id;
    localStorage.setItem('rag_active_conversation', state.activeConversationId);
  }
}

async function createConversation() {
  if (!state.apiKey) {
    setSection('settings');
    showToast('API key required', 'Add a valid key before creating a conversation.', 'danger');
    return;
  }
  try {
    const conversation = await api.createConversation();
    state.activeConversationId = conversation.conversation_id;
    localStorage.setItem('rag_active_conversation', conversation.conversation_id);
    await refreshWorkspace();
    await loadConversation(conversation.conversation_id);
    setSection('conversations');
    showToast('Conversation created', conversation.title || 'New conversation', 'success');
  } catch (error) {
    showToast('Conversation failed', error.message || 'Unable to create conversation.', 'danger');
  }
}

function bootstrapComposerPlaceholderRotation() {
  clearInterval(placeholderTimer);
  placeholderTimer = window.setInterval(() => {
    currentPlaceholderIndex = (currentPlaceholderIndex + 1) % PLACEHOLDERS.length;
    const textarea = document.getElementById('copilot-prompt');
    if (textarea instanceof HTMLTextAreaElement && document.activeElement !== textarea) {
      textarea.placeholder = PLACEHOLDERS[currentPlaceholderIndex];
    }
  }, 4200);
}

function attachResizeHandle() {
  const handle = document.querySelector('[data-resize-handle]');
  if (!handle) {
    return;
  }
  handle.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    dragResize = { startX: event.clientX, startWidth: state.rightWidth };
    handle.setPointerCapture(event.pointerId);
    const onMove = (moveEvent) => {
      if (!dragResize) {
        return;
      }
      const delta = dragResize.startX - moveEvent.clientX;
      setLayoutWidth(dragResize.startWidth + delta);
    };
    const onUp = () => {
      dragResize = null;
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp, { once: true });
  });
}

function attachTextareaAutoGrow() {
  const prompt = document.getElementById('copilot-prompt');
  if (prompt instanceof HTMLTextAreaElement) {
    prompt.style.height = 'auto';
    prompt.style.height = `${Math.max(88, prompt.scrollHeight)}px`;
    prompt.addEventListener('input', () => {
      prompt.style.height = 'auto';
      prompt.style.height = `${Math.max(88, prompt.scrollHeight)}px`;
    });
  }
}

function focusTopbarSearch() {
  const search = document.getElementById('global-search');
  if (search instanceof HTMLInputElement) {
    search.focus();
  }
}

function startApp() {
  makeInitialShell();
  applyLayoutPrefs();
  setThemeMode(state.themeMode);
  bindStaticEvents();
  render();
  bootstrapComposerPlaceholderRotation();
  attachResizeHandle();
  attachTextareaAutoGrow();
  refreshWorkspace();
  loadAndRenderEvaluationSamples();
  requestAnimationFrame(() => {
    if (!state.apiKey) {
      setSection('settings');
      showToast('API key required', 'Open Settings and paste your Bearer token to unlock the workspace.');
    }
  });
}

function selectEvaluation(evaluationJson) {
  try {
    state.selectedEvaluation = typeof evaluationJson === 'string' ? JSON.parse(evaluationJson) : evaluationJson;
    state.activeSection = 'metrics';
    scheduleRender();
  } catch {
    return;
  }
}

function saveSelectedConversationFromActive() {
  setActiveConversationFromDetail();
}

function submitDraftBoard() {
  const draft = state.draftBoard.trim();
  if (!draft) {
    showToast('Draft empty', 'Write something before exporting or copying.');
    return;
  }
  copyText(draft).then(() => showToast('Draft copied', 'Working draft copied to clipboard.', 'success'));
}

function refreshLeftAndCenterAfterMutation() {
  refreshWorkspace();
  scheduleRender();
}

function processCommandPrompt(command) {
  setDraft(command);
  focusComposer();
}

function updateCopilotStatus(text) {
  const status = document.getElementById('copilot-status');
  if (status) {
    status.textContent = text;
  }
}

function clearStreamingState() {
  state.streaming = false;
  updateCopilotStatus('Ready');
  scheduleRender();
}

function dismissUploadState() {
  state.upload = {
    active: false,
    filename: '',
    progress: 0,
    stage: 'Idle',
    status: 'idle',
    error: '',
  };
  scheduleRender();
}

function installObserverHints() {
  const prompt = document.getElementById('copilot-prompt');
  if (prompt instanceof HTMLTextAreaElement) {
    prompt.setAttribute('aria-describedby', 'copilot-help');
  }
}

window.addEventListener('load', () => {
  startApp();
  installObserverHints();
});

// Expose a few helpers for debugging and future integration work.
window.__ragWorkspace = {
  refreshWorkspace,
  loadConversation,
  createConversation,
  submitPrompt,
  setSection,
};
