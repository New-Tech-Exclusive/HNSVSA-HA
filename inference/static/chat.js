/* ═══════════════════════════════════════════════════════════════════
   Arnold Chat — Claude layout · Gemini colors
   ═══════════════════════════════════════════════════════════════════ */

const messagesEl    = document.getElementById('messages');
const messagesInner = document.getElementById('messages-inner');
const welcomeEl     = document.getElementById('welcome');
const inputEl       = document.getElementById('input');
const sendBtn       = document.getElementById('send-btn');
const settingsEl    = document.getElementById('settings-overlay');
const modelInfoEl   = document.getElementById('model-info');
const sidebarEl     = document.getElementById('sidebar');
const sidebarHistory = document.getElementById('sidebar-history');

let isGenerating = false;
let abortController = null;
let chatTitle = null;  // title of the current active chat

// ── Sidebar ──────────────────────────────────────────────────────

function toggleSidebar() {
  sidebarEl.classList.toggle('collapsed');
}

// ── Load model info ──────────────────────────────────────────────

async function loadModelInfo() {
  try {
    const res = await fetch('/api/info');
    const info = await res.json();
    const params = (info.parameters / 1e6).toFixed(0);
    modelInfoEl.textContent = `Arnold ${params}M · d=${info.d_model} · L=${info.num_layers} · ${info.device}`;
  } catch {
    modelInfoEl.textContent = 'Arnold · model loaded';
  }
}
loadModelInfo();

// ── Markdown-lite rendering ──────────────────────────────────────

function renderText(text) {
  // Code blocks
  text = text.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
    return `<pre><code>${escapeHtml(code.trim())}</code></pre>`;
  });
  // Inline code
  text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Bold
  text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic
  text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Line breaks
  text = text.replace(/\n/g, '<br>');
  return text;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ── Message helpers ──────────────────────────────────────────────

function addMessage(role, html) {
  welcomeEl.style.display = 'none';

  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${role}`;

  const avatarLabel = role === 'user' ? 'U' : 'A';

  msgDiv.innerHTML = `
    <div class="message-content">
      <div class="avatar">${avatarLabel}</div>
      <div class="bubble">${html}</div>
    </div>
  `;

  messagesInner.appendChild(msgDiv);
  scrollToBottom();
  return msgDiv;
}

function createReasoningPanel(parentEl) {
  const panel = document.createElement('details');
  panel.className = 'reasoning-panel';
  panel.open = false;

  const summary = document.createElement('summary');
  summary.className = 'reasoning-summary';
  summary.textContent = 'Show reasoning';

  const body = document.createElement('div');
  body.className = 'reasoning-body';
  body.textContent = 'Reasoning will appear here if provided by the model.';

  panel.appendChild(summary);
  panel.appendChild(body);
  parentEl.appendChild(panel);

  return { panel, summary, body };
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ── Auto-resize textarea ────────────────────────────────────────

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (isGenerating) {
      stopGeneration();
    } else {
      sendMessage();
    }
  }
}

// ── Settings ────────────────────────────────────────────────────

function toggleSettings() {
  settingsEl.classList.toggle('open');
}

function closeSettingsOutside(e) {
  if (e.target === settingsEl) toggleSettings();
}

// ── Get sampling params ─────────────────────────────────────────

function getParams() {
  return {
    temperature: parseFloat(document.getElementById('temperature').value),
    top_k: parseInt(document.getElementById('top_k').value),
    top_p: parseFloat(document.getElementById('top_p').value),
    max_tokens: parseInt(document.getElementById('max_tokens').value),
    repetition_penalty: parseFloat(document.getElementById('repetition_penalty').value),
  };
}

// ── Clear chat ──────────────────────────────────────────────────

function clearChat() {
  // Archive current conversation title to sidebar history
  if (chatTitle && messagesInner.children.length > 0) {
    const item = document.createElement('div');
    item.className = 'history-item';
    item.textContent = chatTitle;
    item.title = chatTitle;
    sidebarHistory.prepend(item);
  }
  chatTitle = null;
  messagesInner.innerHTML = '';
  welcomeEl.style.display = '';
}

// ── Suggestion click ────────────────────────────────────────────

function sendSuggestion(el) {
  const span = el.querySelector('span');
  inputEl.value = span ? span.textContent : el.textContent;
  sendMessage();
}

// ── Stop generation ─────────────────────────────────────────────

function stopGeneration() {
  if (abortController) {
    abortController.abort();
    abortController = null;
  }
}

// ── Send message ────────────────────────────────────────────────

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || isGenerating) return;

  // Reset input
  inputEl.value = '';
  inputEl.style.height = 'auto';

  // Set chat title from first message
  if (!chatTitle) chatTitle = text.slice(0, 60) + (text.length > 60 ? '…' : '');

  // Add user message
  addMessage('user', escapeHtml(text));

  // Add assistant placeholder
  const assistantMsg = addMessage('assistant',
    '<div class="typing-indicator"><span></span><span></span><span></span></div>');
  const bubbleEl = assistantMsg.querySelector('.bubble');
  const {
    panel: reasoningPanelEl,
    summary: reasoningSummaryEl,
    body: reasoningBodyEl,
  } = createReasoningPanel(bubbleEl);
  reasoningSummaryEl.addEventListener('click', () => {
    setTimeout(() => {
      reasoningSummaryEl.textContent = reasoningPanelEl.open ? 'Hide reasoning' : 'Show reasoning';
    }, 0);
  });

  const assistantTextEl = document.createElement('div');
  assistantTextEl.className = 'assistant-text';

  const statsEl = document.createElement('div');
  statsEl.className = 'stats';
  statsEl.style.display = 'none';

  // Update button to stop
  isGenerating = true;
  sendBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>';
  sendBtn.classList.add('stop-btn');
  sendBtn.onclick = stopGeneration;

  let fullText = '';
  let reasoningText = '';
  let sawReasoning = false;
  abortController = new AbortController();

  try {
    const params = getParams();
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, ...params }),
      signal: abortController.signal,
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    // Clear typing indicator on first token
    let firstToken = true;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // Keep incomplete line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = JSON.parse(line.slice(6));

        if (data.token) {
          if (firstToken) {
            bubbleEl.innerHTML = '';
            bubbleEl.appendChild(reasoningPanelEl);
            bubbleEl.appendChild(assistantTextEl);
            bubbleEl.appendChild(statsEl);
            firstToken = false;
          }
          fullText += data.token;
          assistantTextEl.innerHTML = renderText(fullText);
          scrollToBottom();
        }

        if (data.reasoning || data.thinking) {
          sawReasoning = true;
          const chunk = data.reasoning || data.thinking;
          reasoningText += chunk;
          reasoningBodyEl.innerHTML = renderText(reasoningText);
          reasoningSummaryEl.textContent = 'Hide reasoning';
        }

        if (data.done) {
          // Append stats
          if (firstToken) {
            bubbleEl.innerHTML = '';
            bubbleEl.appendChild(reasoningPanelEl);
            bubbleEl.appendChild(assistantTextEl);
            bubbleEl.appendChild(statsEl);
            assistantTextEl.innerHTML = renderText(fullText);
            firstToken = false;
          }
          statsEl.style.display = '';
          statsEl.textContent = `${data.tokens} tokens · ${data.tok_per_s} tok/s · ${data.time_s}s`;
          if (!sawReasoning) {
            reasoningBodyEl.textContent = 'No reasoning trace was provided by the backend for this response.';
          }
          scrollToBottom();
        }

        if (data.error) {
          if (firstToken) {
            bubbleEl.innerHTML = '';
            bubbleEl.appendChild(reasoningPanelEl);
            bubbleEl.appendChild(assistantTextEl);
            bubbleEl.appendChild(statsEl);
            firstToken = false;
          }
          assistantTextEl.innerHTML = `<span style="color:#e55565">${escapeHtml(data.error)}</span>`;
          if (!sawReasoning) {
            reasoningBodyEl.textContent = 'No reasoning trace was provided due to a generation error.';
          }
        }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      if (fullText) {
        bubbleEl.innerHTML = '';
        bubbleEl.appendChild(reasoningPanelEl);
        bubbleEl.appendChild(assistantTextEl);
        bubbleEl.appendChild(statsEl);
        assistantTextEl.innerHTML = renderText(fullText);
        statsEl.style.display = '';
        statsEl.style.color = '#e55565';
        statsEl.textContent = 'Generation stopped';
        if (!sawReasoning) {
          reasoningBodyEl.textContent = 'No reasoning trace was provided before generation stopped.';
        }
      } else {
        bubbleEl.innerHTML = '';
        bubbleEl.appendChild(reasoningPanelEl);
        bubbleEl.appendChild(assistantTextEl);
        bubbleEl.appendChild(statsEl);
        assistantTextEl.innerHTML = '<span style="color:var(--text-muted)">Stopped</span>';
        if (!sawReasoning) {
          reasoningBodyEl.textContent = 'No reasoning trace was provided before generation stopped.';
        }
      }
    } else {
      bubbleEl.innerHTML = '';
      bubbleEl.appendChild(reasoningPanelEl);
      bubbleEl.appendChild(assistantTextEl);
      bubbleEl.appendChild(statsEl);
      assistantTextEl.innerHTML = `<span style="color:#e55565">Error: ${escapeHtml(err.message)}</span>`;
      if (!sawReasoning) {
        reasoningBodyEl.textContent = 'No reasoning trace was provided due to an error during generation.';
      }
    }
  } finally {
    isGenerating = false;
    abortController = null;
    sendBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>';
    sendBtn.classList.remove('stop-btn');
    sendBtn.onclick = sendMessage;
    inputEl.focus();
  }
}

// Focus input on load
inputEl.focus();
