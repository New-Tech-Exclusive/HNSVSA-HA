/* ═══════════════════════════════════════════════════════════════════
   Arnold Chat — Streaming SSE chat logic
   ═══════════════════════════════════════════════════════════════════ */

const messagesEl    = document.getElementById('messages');
const messagesInner = document.getElementById('messages-inner');
const welcomeEl     = document.getElementById('welcome');
const inputEl       = document.getElementById('input');
const sendBtn       = document.getElementById('send-btn');
const settingsEl    = document.getElementById('settings-overlay');
const modelInfoEl   = document.getElementById('model-info');

let isGenerating = false;
let abortController = null;

// ── Load model info ──────────────────────────────────────────────

async function loadModelInfo() {
  try {
    const res = await fetch('/api/info');
    const info = await res.json();
    const params = (info.parameters / 1e6).toFixed(0);
    modelInfoEl.textContent = `${params}M params · d=${info.d_model} · L=${info.num_layers} · ${info.device}`;
  } catch {
    modelInfoEl.textContent = 'Model loaded';
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
  messagesInner.innerHTML = '';
  welcomeEl.style.display = '';
}

// ── Suggestion click ────────────────────────────────────────────

function sendSuggestion(el) {
  inputEl.value = el.textContent;
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

  // Add user message
  addMessage('user', escapeHtml(text));

  // Add assistant placeholder
  const assistantMsg = addMessage('assistant',
    '<div class="typing-indicator"><span></span><span></span><span></span></div>');
  const bubbleEl = assistantMsg.querySelector('.bubble');

  // Update button to stop
  isGenerating = true;
  sendBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>';
  sendBtn.classList.add('stop-btn');
  sendBtn.onclick = stopGeneration;

  let fullText = '';
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
            firstToken = false;
          }
          fullText += data.token;
          bubbleEl.innerHTML = renderText(fullText);
          scrollToBottom();
        }

        if (data.done) {
          // Append stats
          const statsHtml = `<div class="stats">${data.tokens} tokens · ${data.tok_per_s} tok/s · ${data.time_s}s</div>`;
          bubbleEl.innerHTML = renderText(fullText) + statsHtml;
          scrollToBottom();
        }

        if (data.error) {
          bubbleEl.innerHTML = `<span style="color:#e55565">${escapeHtml(data.error)}</span>`;
        }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      if (fullText) {
        bubbleEl.innerHTML = renderText(fullText) +
          '<div class="stats" style="color:#e55565">Generation stopped</div>';
      } else {
        bubbleEl.innerHTML = '<span style="color:var(--text-muted)">Stopped</span>';
      }
    } else {
      bubbleEl.innerHTML = `<span style="color:#e55565">Error: ${escapeHtml(err.message)}</span>`;
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
