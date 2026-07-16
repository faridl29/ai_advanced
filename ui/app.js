document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const modelSelect = document.getElementById('model-select');
    const ragDot = document.getElementById('rag-dot');
    const indexedCount = document.getElementById('indexed-count');
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const uploadProgressContainer = document.getElementById('upload-progress-container');
    const uploadProgressFill = document.getElementById('upload-progress-fill');
    const uploadProgressText = document.getElementById('upload-progress-text');
    const uploadStatusMsg = document.getElementById('upload-status-msg');
    
    const chatMessages = document.getElementById('chat-messages');
    const emptyState = document.getElementById('empty-state');
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const btnSend = document.getElementById('btn-send');
    const clearChatBtn = document.getElementById('clear-chat-btn');
    
    const modelPill = document.getElementById('model-pill');
    const intentPill = document.getElementById('intent-pill');
    const latencyVal = document.getElementById('latency-val');

    // State Variables
    let currentSessionId = localStorage.getItem('chat_session_id') || `session-${Date.now()}`;
    localStorage.setItem('chat_session_id', currentSessionId);
    
    let chatHistory = JSON.parse(localStorage.getItem(`chat_history_${currentSessionId}`)) || [];
    let isGenerating = false;

    // Initialize Lucide Icons
    lucide.createIcons();

    // Configure Marked Options
    marked.setOptions({
        gfm: true,
        breaks: true,
        headerIds: false,
        mangle: false
    });

    // Load available models
    async function loadModels() {
        try {
            const res = await fetch('/v1/models');
            if (!res.ok) throw new Error('Failed to load models');
            const data = await res.json();
            
            // Filter out embedding models
            const models = (data.data || [])
                .map(m => m.id || m.name)
                .filter(name => name && !name.toLowerCase().includes('embed'));
            
            modelSelect.innerHTML = '';
            if (models.length === 0) {
                modelSelect.innerHTML = '<option value="qwen3:4b">qwen3:4b (Fallback)</option>';
            } else {
                models.forEach(model => {
                    const opt = document.createElement('option');
                    opt.value = model;
                    opt.textContent = model;
                    modelSelect.appendChild(opt);
                });
            }
            
            // Set saved model preference
            const savedModel = localStorage.getItem('selected_model');
            if (savedModel && modelSelect.querySelector(`option[value="${savedModel}"]`)) {
                modelSelect.value = savedModel;
            }
            
            updateHeaderModelPill();
        } catch (err) {
            console.error(err);
            modelSelect.innerHTML = '<option value="qwen3:4b">qwen3:4b (Fallback)</option>';
            updateHeaderModelPill();
        }
    }

    // Load RAG Ingestion Status
    async function loadRagStatus() {
        try {
            const res = await fetch('/v1/rag/status');
            if (!res.ok) throw new Error('Qdrant offline');
            const data = await res.json();
            
            if (data.status === 'ok') {
                ragDot.className = 'status-dot status-online';
                indexedCount.textContent = `${data.documents_indexed || 0} docs`;
            } else {
                throw new Error('Qdrant in error state');
            }
        } catch (err) {
            console.error(err);
            ragDot.className = 'status-dot status-offline';
            indexedCount.textContent = 'Offline';
        }
    }

    // Update Model Pill in Header
    function updateHeaderModelPill() {
        modelPill.textContent = modelSelect.value || 'qwen3:4b';
    }

    modelSelect.addEventListener('change', () => {
        localStorage.setItem('selected_model', modelSelect.value);
        updateHeaderModelPill();
    });

    // File Ingestion drag-and-drop
    uploadZone.addEventListener('click', () => fileInput.click());
    
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-active');
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-active');
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-active');
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFileUpload(fileInput.files);
        }
    });

    async function handleFileUpload(files) {
        uploadStatusMsg.className = 'upload-status-msg';
        uploadStatusMsg.textContent = '';
        uploadProgressContainer.style.display = 'block';
        uploadProgressFill.style.width = '0%';
        
        let completed = 0;
        let errors = 0;

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            uploadProgressText.textContent = `Uploading ${file.name} (${i+1}/${files.length})...`;
            
            const formData = new FormData();
            formData.append('file', file);

            try {
                // Simulate progress fill to 70% during request
                uploadProgressFill.style.width = '70%';
                
                const res = await fetch('/v1/rag/ingest', {
                    method: 'POST',
                    body: formData
                });
                
                if (!res.ok) throw new Error('Ingestion failed');
                
                completed++;
            } catch (err) {
                console.error(err);
                errors++;
            }
            
            // Update progress calculation
            const pct = Math.round(((i + 1) / files.length) * 100);
            uploadProgressFill.style.width = `${pct}%`;
        }

        setTimeout(() => {
            uploadProgressContainer.style.display = 'none';
            if (errors === 0) {
                uploadStatusMsg.className = 'upload-status-msg success';
                uploadStatusMsg.textContent = `Ingested ${completed} files successfully!`;
            } else {
                uploadStatusMsg.className = 'upload-status-msg error';
                uploadStatusMsg.textContent = `Completed ${completed} uploads, ${errors} failed.`;
            }
            loadRagStatus();
        }, 800);
    }

    // Render message log
    function renderHistory() {
        // Clear all except empty state
        const messages = chatMessages.querySelectorAll('.chat-row, .tool-log');
        messages.forEach(m => m.remove());

        if (chatHistory.length === 0) {
            emptyState.style.display = 'flex';
        } else {
            emptyState.style.display = 'none';
            chatHistory.forEach(msg => {
                appendBubble(msg.role, msg.content, msg.thinking, msg.sources);
            });
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }

    // Helper to parse thinking & final answer
    function parseContent(text) {
        let thinking = "";
        let answer = "";
        
        const thinkStart = text.indexOf("<think>");
        const thinkEnd = text.indexOf("</think>");
        
        if (thinkStart !== -1) {
            if (thinkEnd !== -1) {
                thinking = text.substring(thinkStart + 7, thinkEnd);
                answer = text.substring(0, thinkStart) + text.substring(thinkEnd + 8);
            } else {
                thinking = text.substring(thinkStart + 7);
                answer = text.substring(0, thinkStart);
            }
        } else {
            if (thinkEnd !== -1) {
                thinking = text.substring(0, thinkEnd);
                answer = text.substring(thinkEnd + 8);
            } else {
                thinking = "";
                answer = text;
            }
        }
        return { thinking: thinking.trim(), answer: answer.trim() };
    }

    // Create a chat bubble DOM element
    function appendBubble(role, content, thinking = "", sources = [], id = null) {
        const row = document.createElement('div');
        row.className = `chat-row ${role}`;
        if (id) row.id = id;

        const bubble = document.createElement('div');
        bubble.className = `bubble bubble-${role}`;

        const bubbleContent = document.createElement('div');
        bubbleContent.className = 'bubble-content';

        let innerHTML = '';

        // If there's thinking, render collapsible accordion
        if (thinking) {
            innerHTML += `
            <details class="reasoning" open>
                <summary>Reasoning Process</summary>
                <div class="reasoning-content">${escapeHTML(thinking)}</div>
            </details>`;
        }

        // Parse markdown content
        if (content) {
            innerHTML += `<div class="markdown-body">${marked.parse(content)}</div>`;
        }

        // If RAG sources are present, render them
        if (sources && sources.length > 0) {
            const uniqueSources = [...new Set(sources)];
            innerHTML += '<div class="sources-container">';
            uniqueSources.forEach(src => {
                innerHTML += `<span class="source-chip"><i data-lucide="file-text" class="icon-small"></i> ${escapeHTML(src)}</span>`;
            });
            innerHTML += '</div>';
        }

        bubbleContent.innerHTML = innerHTML;
        bubble.appendChild(bubbleContent);
        row.appendChild(bubble);
        chatMessages.appendChild(row);

        // Render icons
        lucide.createIcons({ attrs: { class: 'icon-small' } });
        
        return row;
    }

    function escapeHTML(str) {
        return str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    // Append Tool execution message logs
    function appendToolLog(text) {
        const log = document.createElement('div');
        log.className = 'tool-log';
        log.innerHTML = `<i data-lucide="cpu"></i> <span>${escapeHTML(text)}</span>`;
        chatMessages.appendChild(log);
        lucide.createIcons();
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return log;
    }

    // Show typing bounce indicator
    function showTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'chat-row assistant typing-row';
        indicator.innerHTML = `
            <div class="bubble bubble-assistant">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>`;
        chatMessages.appendChild(indicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function removeTypingIndicator() {
        const row = chatMessages.querySelector('.typing-row');
        if (row) row.remove();
    }

    // Submit a chat message
    async function submitMessage(messageText) {
        if (!messageText || isGenerating) return;
        
        isGenerating = true;
        emptyState.style.display = 'none';
        
        // Append user bubble
        appendBubble('user', messageText);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Build history array formatted for Backend UnifiedChatRequest
        const apiHistory = chatHistory.map(h => ({
            role: h.role,
            content: h.thinking ? `<think>${h.thinking}</think>\n${h.content}` : h.content
        }));

        chatHistory.push({ role: 'user', content: messageText });
        localStorage.setItem(`chat_history_${currentSessionId}`, JSON.stringify(chatHistory));

        showTypingIndicator();
        btnSend.disabled = true;

        // Streaming states
        let accumulatedText = "";
        let streamSources = [];
        let assistantBubbleId = `assistant-bubble-${Date.now()}`;
        let bubbleAdded = false;

        try {
            const res = await fetch('/v1/chat/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: messageText,
                    model: modelSelect.value,
                    history: apiHistory
                })
            });

            if (!res.ok) throw new Error('API server returned error');

            const reader = res.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';

            removeTypingIndicator();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // save incomplete line

                for (const line of lines) {
                    if (line.trim() === '') continue;
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            
                            if (data.event === 'metadata') {
                                if (data.intent) updateIntentPill(data.intent);
                                if (data.model_used) modelPill.textContent = data.model_used;
                            } 
                            else if (data.event === 'tool_start') {
                                appendToolLog(`Running tool: ${data.tool} (${JSON.stringify(data.args || {})})`);
                            } 
                            else if (data.event === 'tool_end') {
                                // optional update
                            } 
                            else if (data.event === 'content') {
                                accumulatedText += data.delta || '';
                                
                                const parsed = parseContent(accumulatedText);
                                
                                if (!bubbleAdded) {
                                    appendBubble('assistant', parsed.answer, parsed.thinking, [], assistantBubbleId);
                                    bubbleAdded = true;
                                } else {
                                    updateBubble(assistantBubbleId, parsed.answer, parsed.thinking);
                                }
                                chatMessages.scrollTop = chatMessages.scrollHeight;
                            } 
                            else if (data.event === 'done') {
                                updateIntentPill(data.intent);
                                if (data.latency_ms) latencyVal.textContent = `${data.latency_ms.toFixed(1)}ms`;
                                if (data.sources) streamSources = data.sources;
                                
                                const parsedDone = parseContent(data.answer || accumulatedText);
                                const parsedAccum = parseContent(accumulatedText);
                                const finalThinking = parsedDone.thinking || parsedAccum.thinking;
                                const finalAnswer = parsedDone.answer;
                                
                                if (!bubbleAdded) {
                                    appendBubble('assistant', finalAnswer, finalThinking, streamSources, assistantBubbleId);
                                    bubbleAdded = true;
                                } else {
                                    updateBubble(assistantBubbleId, finalAnswer, finalThinking, streamSources);
                                }
                                
                                // Save assistant turn in history
                                chatHistory.push({
                                    role: 'assistant',
                                    content: finalAnswer,
                                    thinking: finalThinking,
                                    sources: streamSources
                                });
                                localStorage.setItem(`chat_history_${currentSessionId}`, JSON.stringify(chatHistory));
                            }
                            else if (data.event === 'error') {
                                throw new Error(data.detail || 'Streaming execution error');
                            }
                        } catch (jsonErr) {
                            console.error('Error parsing SSE line:', jsonErr, line);
                        }
                    }
                }
            }
        } catch (err) {
            console.error(err);
            removeTypingIndicator();
            appendBubble('assistant', `⚠️ **Error**: ${err.message || 'Connection lost'}`);
        } finally {
            isGenerating = false;
            btnSend.disabled = false;
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }

    function updateBubble(id, content, thinking = "", sources = []) {
        const row = document.getElementById(id);
        if (!row) return;

        const bubbleContent = row.querySelector('.bubble-content');
        if (!bubbleContent) return;

        let innerHTML = '';

        if (thinking) {
            // Check if accordion exists to keep its open/closed state
            const details = bubbleContent.querySelector('details.reasoning');
            const isOpen = details ? details.open : true;
            
            innerHTML += `
            <details class="reasoning" ${isOpen ? 'open' : ''}>
                <summary>Reasoning Process</summary>
                <div class="reasoning-content">${escapeHTML(thinking)}</div>
            </details>`;
        }

        if (content) {
            innerHTML += `<div class="markdown-body">${marked.parse(content)}</div>`;
        }

        if (sources && sources.length > 0) {
            const uniqueSources = [...new Set(sources)];
            innerHTML += '<div class="sources-container">';
            uniqueSources.forEach(src => {
                innerHTML += `<span class="source-chip"><i data-lucide="file-text" class="icon-small"></i> ${escapeHTML(src)}</span>`;
            });
            innerHTML += '</div>';
        }

        bubbleContent.innerHTML = innerHTML;
        lucide.createIcons({ attrs: { class: 'icon-small' } });
    }

    function updateIntentPill(intent) {
        if (!intent) return;
        intentPill.textContent = intent.toUpperCase().replace('_', ' ');
        intentPill.className = 'pill';
        
        switch(intent.toLowerCase()) {
            case 'direct_chat':
                intentPill.classList.add('pill-success');
                break;
            case 'rag_query':
                intentPill.classList.add('pill-info');
                break;
            case 'agent_task':
                intentPill.classList.add('pill-purple');
                break;
            case 'blocked':
                intentPill.classList.add('pill-error');
                break;
            default:
                intentPill.classList.add('pill-warning');
        }
    }

    // Auto-growing textarea
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = `${chatInput.scrollHeight}px`;
    });

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });

    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const text = chatInput.value.trim();
        if (text) {
            chatInput.value = '';
            chatInput.style.height = 'auto';
            submitMessage(text);
        }
    });

    // Quick action clicks
    document.querySelectorAll('.qa-card').forEach(card => {
        card.addEventListener('click', () => {
            const prompt = card.getAttribute('data-prompt');
            if (prompt) submitMessage(prompt);
        });
    });

    // Clear history
    clearChatBtn.addEventListener('click', () => {
        if (confirm('Are you sure you want to clear this conversation?')) {
            chatHistory = [];
            localStorage.removeItem(`chat_history_${currentSessionId}`);
            renderHistory();
            latencyVal.textContent = '0.0ms';
            updateIntentPill('direct_chat');
        }
    });

    // Run Initial Loaders
    loadModels();
    loadRagStatus();
    renderHistory();

    // Check status every 15s
    setInterval(loadRagStatus, 15000);
});
