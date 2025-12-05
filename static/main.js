document.addEventListener('DOMContentLoaded', () => {
    const socket = io();
    window.socket = socket;
    const videoStream = document.getElementById('video-stream');
    const graphStream = document.getElementById('graph-stream');
    const thermalStream = document.getElementById('thermal-stream');
    const statusElement = document.getElementById('button-status');
    const configButton = document.querySelector('.btnConfig');
    const overlay = document.getElementById('config-overlay');
    const form = document.getElementById('config-form');
    const timeInput = document.getElementById('config-time');
    const rotationInput = document.getElementById('config-rotation');
    const scheduledTime = document.getElementById('scheduled-time');
    const scheduledRotation = document.getElementById('scheduled-rotation');
    const remainingTime = document.getElementById('remaining-time');
    const feedback = document.getElementById('config-feedback');
    const autoFeedToggle = document.getElementById('auto-feed-enabled');
    const autoFeedStatus = document.getElementById('auto-feed-status');
    const autoFeedLegend = document.getElementById('auto-feed-legend');
    const runMotorButton = document.getElementById('run-motor-now');
    const sensorsMetric = document.getElementById('metric-sensors-active');
    const latencyMetric = document.getElementById('metric-latency');
    const lastCaptureMetric = document.getElementById('metric-last-capture');
    const videoFeeds = Array.from(document.querySelectorAll('.video-feed'));
    const METRIC_STORAGE_KEY = 'argus.metrics';
    const DEFAULT_METRICS = {
        sensorsActive: 0,
        latencyMs: null,
        lastCapture: null,
        timestamp: null
    };
    let metricsState = { ...DEFAULT_METRICS };
    let monitorInterval = null;
    let countdownInterval = null;
    let currentAutoFeedEnabled = true;

    const loadStoredMetrics = () => {
        try {
            const raw = localStorage.getItem(METRIC_STORAGE_KEY);
            if (!raw) return;
            const parsed = JSON.parse(raw);
            if (parsed && typeof parsed === 'object') {
                metricsState = { ...DEFAULT_METRICS, ...parsed };
            }
        } catch (error) {
            console.warn('Não foi possível carregar métricas armazenadas localmente:', error);
        }
    };

    const saveMetrics = () => {
        try {
            localStorage.setItem(METRIC_STORAGE_KEY, JSON.stringify(metricsState));
        } catch (error) {
            console.warn('Não foi possível salvar métricas localmente:', error);
        }
    };

    const formatLatency = (latencyMs) => {
        if (latencyMs === null || latencyMs === undefined || Number.isNaN(latencyMs)) {
            return '—';
        }
        if (latencyMs >= 1000) {
            return `${(latencyMs / 1000).toFixed(2)} s`;
        }
        return `${Math.round(latencyMs)} ms`;
    };

    const formatRelativeTime = (timestamp) => {
        if (!timestamp) return '—';
        const delta = Date.now() - timestamp;
        if (delta < 0) return 'agora mesmo';

        const seconds = Math.floor(delta / 1000);
        if (seconds < 5) return 'agora mesmo';
        if (seconds < 60) return `${seconds}s atrás`;

        const minutes = Math.floor(seconds / 60);
        if (minutes < 60) return `${minutes}m atrás`;

        const hours = Math.floor(minutes / 60);
        if (hours < 24) return `${hours}h atrás`;

        const days = Math.floor(hours / 24);
        return `${days}d atrás`;
    };

    const updateMetricsUI = () => {
        if (sensorsMetric) {
            sensorsMetric.textContent = String(metricsState.sensorsActive).padStart(2, '0');
        }
        if (latencyMetric) {
            latencyMetric.textContent = formatLatency(metricsState.latencyMs);
        }
        if (lastCaptureMetric) {
            lastCaptureMetric.textContent = formatRelativeTime(metricsState.lastCapture);
        }
    };

    const handleSocketConnection = () => {
        console.log('Conectado ao servidor via WebSocket!');
        // Quando conectarmos novamente, podemos considerar cancelar a atualização automática.
    };

    socket.on('connect', handleSocketConnection);

    const createStreamHandler = (target, streamId = null) => {
        if (!target) {
            return () => {};
        }

        let pendingFrame = null;
        let rafId = null;

        const applyFrame = () => {
            if (!pendingFrame) {
                rafId = null;
                return;
            }

            target.src = pendingFrame;
            pendingFrame = null;
            rafId = null;
        };

        return (payload) => {
            pendingFrame = `data:image/jpeg;base64,${payload.image}`;

            if (streamId === 'video') {
                const now = Date.now();
                let latencyMs = metricsState.latencyMs;

                if (payload.sent_at !== undefined && payload.sent_at !== null) {
                    const sentAtMs = Number(payload.sent_at) * 1000;
                    if (Number.isFinite(sentAtMs)) {
                        latencyMs = Math.max(0, now - sentAtMs);
                    }
                }

                metricsState = {
                    ...metricsState,
                    sensorsActive: Math.max(1, Number(metricsState.sensorsActive) || 0),
                    latencyMs,
                    lastCapture: now,
                    timestamp: now
                };
                saveMetrics();
                updateMetricsUI();
            }

            if (rafId === null) {
                rafId = requestAnimationFrame(applyFrame);
            }
        };
    };

    const handleVideoFrame = createStreamHandler(videoStream, 'video');
    const handleGraphFrame = createStreamHandler(graphStream, 'graph');
    const handleThermalFrame = createStreamHandler(thermalStream, 'thermal');

    socket.on('video_frame', handleVideoFrame);
    socket.on('graph_frame', handleGraphFrame);
    socket.on('thermal_frame', handleThermalFrame);

    socket.on('button_status', function(msg) {
        console.log('Status do bebedouro recebido: ' + msg.data);

        if (msg.data.includes("DRINKING")) {
            statusElement.textContent = 'BEBENDO';
            statusElement.classList.remove('status-off');
            statusElement.classList.add('status-on');
        } else {
            statusElement.textContent = 'PARADO';
            statusElement.classList.remove('status-on');
            statusElement.classList.add('status-off');
        }
    });

    const stopCountdown = () => {
        if (countdownInterval) {
            clearInterval(countdownInterval);
            countdownInterval = null;
        }
    };

    const closeModal = () => {
        if (!overlay) return;
        overlay.setAttribute('hidden', '');
        overlay.classList.remove('visible');
        stopCountdown();
        if (feedback) {
            feedback.textContent = '';
            feedback.classList.remove('error', 'success');
        }
    };

    const formatTimeDisplay = (hour, minute) => {
        const h = String(hour).padStart(2, '0');
        const m = String(minute).padStart(2, '0');
        return `${h}:${m}`;
    };

    const formatDuration = (seconds) => {
        if (seconds === null || seconds === undefined) {
            return '—';
        }

        const safeSeconds = Math.max(0, seconds);
        const hrs = Math.floor(safeSeconds / 3600);
        const mins = Math.floor((safeSeconds % 3600) / 60);
        const secs = safeSeconds % 60;

        return `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    };

    const updateCountdown = (seconds) => {
        if (remainingTime) {
            remainingTime.textContent = formatDuration(seconds);
        }
    };

    const startCountdown = (seconds) => {
        if (countdownInterval) {
            clearInterval(countdownInterval);
        }

        let current = typeof seconds === 'number' ? seconds : null;
        updateCountdown(current);

        if (current === null) {
            return;
        }

        countdownInterval = setInterval(() => {
            current = Math.max(0, current - 1);
            updateCountdown(current);
            if (current <= 0) {
                clearInterval(countdownInterval);
                countdownInterval = null;
            }
        }, 1000);
    };

    const applyAutoFeedState = (enabled) => {
        currentAutoFeedEnabled = Boolean(enabled);
        const statusText = currentAutoFeedEnabled ? 'Ativa' : 'Desativada';

        if (autoFeedToggle) {
            autoFeedToggle.checked = currentAutoFeedEnabled;
        }

        if (autoFeedStatus) {
            autoFeedStatus.textContent = statusText;
            autoFeedStatus.classList.toggle('is-active', currentAutoFeedEnabled);
            autoFeedStatus.classList.toggle('is-inactive', !currentAutoFeedEnabled);
        }

        if (autoFeedLegend) {
            autoFeedLegend.textContent = statusText;
            autoFeedLegend.classList.toggle('is-active', currentAutoFeedEnabled);
            autoFeedLegend.classList.toggle('is-inactive', !currentAutoFeedEnabled);
        }

        if (!currentAutoFeedEnabled) {
            stopCountdown();
            updateCountdown(null);
        }
    };

    const populateSchedule = (data) => {
        if (!data) return;
        const { hora, minuto, rotacao, tempo_restante_segundos, auto_feed_enabled } = data;

        if (scheduledTime && hora !== undefined && minuto !== undefined) {
            scheduledTime.textContent = formatTimeDisplay(hora, minuto);
            if (timeInput) {
                timeInput.value = formatTimeDisplay(hora, minuto);
            }
        }

        if (scheduledRotation && rotacao !== undefined) {
            const numericRotation = Number(rotacao);
            const formattedRotation = Number.isFinite(numericRotation)
                ? numericRotation.toLocaleString('pt-BR', { maximumFractionDigits: 2 })
                : rotacao;

            scheduledRotation.textContent = `${formattedRotation}°`;
        return (payload) => {
            pendingFrame = `data:image/jpeg;base64,${payload.image}`;

            if (streamId === 'video') {
                const now = Date.now();
                let latencyMs = metricsState.latencyMs;

                if (payload.sent_at !== undefined && payload.sent_at !== null) {
                    const sentAtMs = Number(payload.sent_at) * 1000;
                    if (Number.isFinite(sentAtMs)) {
                        latencyMs = Math.max(0, now - sentAtMs);
                    }
                }

                metricsState = {
                    ...metricsState,
                    sensorsActive: Math.max(1, Number(metricsState.sensorsActive) || 0),
                    latencyMs,
                    lastCapture: now,
                    timestamp: now
                };
                saveMetrics();
                updateMetricsUI();
            }
            }
        }

        applyAutoFeedState(auto_feed_enabled !== undefined ? auto_feed_enabled : true);

        if (currentAutoFeedEnabled && typeof tempo_restante_segundos === 'number') {
        const handleVideoFrame = createStreamHandler(videoStream, 'video');
        const handleGraphFrame = createStreamHandler(graphStream, 'graph');
        const handleThermalFrame = createStreamHandler(thermalStream, 'thermal');
        } else {
            stopCountdown();
            updateCountdown(null);
        }
    };

    const loadSchedule = async () => {
        try {
            const response = await fetch('/api/motor/schedule');
            if (!response.ok) {
                throw new Error('Falha ao obter o agendamento.');
            }
            const data = await response.json();
            populateSchedule(data);
        } catch (error) {
            console.error(error);
            if (feedback) {
                feedback.textContent = 'Não foi possível carregar o agendamento atual.';
                feedback.classList.add('error');
            }
        }
    };

    const openModal = async () => {
        if (!overlay) return;
        overlay.removeAttribute('hidden');
        overlay.classList.add('visible');
        await loadSchedule();
    };

    if (configButton) {
        configButton.addEventListener('click', () => {
            openModal();
        });
    }

    if (autoFeedToggle) {
        autoFeedToggle.addEventListener('change', () => {
            applyAutoFeedState(autoFeedToggle.checked);
        });
    }

    if (overlay) {
        overlay.addEventListener('click', (event) => {
            const trigger = event.target.closest('[data-close-modal]');
            if (trigger) {
                closeModal();
            }
        });
    }

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && !overlay?.hasAttribute('hidden')) {
            closeModal();
        }
    });

    const probeStream = async (url) => {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);

        try {
            const start = performance.now();
            const response = await fetch(url, {
                method: 'GET',
                cache: 'no-store',
                signal: controller.signal,
            });

            if (!response.ok) {
                return { ok: false, latency: null };
            }

            if (!response.body || !response.body.getReader) {
                return { ok: true, latency: performance.now() - start };
            }

            const reader = response.body.getReader();
            try {
                await reader.read();
            } catch (readError) {
                // Ignore read errors triggered by abort.
            }

            const latency = performance.now() - start;
            controller.abort();
            return { ok: true, latency };
        } catch (error) {
            return { ok: false, latency: null };
        } finally {
            clearTimeout(timeoutId);
        }
    };

    const computeMetrics = async () => {
        if (!videoFeeds.length) {
            metricsState = { ...metricsState, sensorsActive: 0 };
            updateMetricsUI();
            return;
        }

        let activeCount = 0;
        let latencySum = 0;
        let latencyCount = 0;
        let latestCapture = metricsState.lastCapture;

        const checks = videoFeeds.map(async (feed) => {
            const currentSrc = feed.src;
            if (!currentSrc) {
                return;
            }

            if (currentSrc.startsWith('data:')) {
                activeCount += 1;
                latestCapture = Date.now();
                return;
            }

            const probeUrl = new URL(currentSrc, window.location.origin);
            probeUrl.searchParams.set('sensorPing', Date.now().toString());
            const result = await probeStream(probeUrl.toString());
            if (result.ok) {
                activeCount += 1;
                if (result.latency !== null) {
                    latencySum += result.latency;
                    latencyCount += 1;
                }
                latestCapture = Date.now();
            }
        });

        await Promise.all(checks);

        const now = Date.now();
        const averagedLatency = latencyCount > 0 ? latencySum / latencyCount : metricsState.latencyMs;
        const resolvedCapture = latestCapture || metricsState.lastCapture;
        const resolvedActive = activeCount || metricsState.sensorsActive || 0;

        metricsState = {
            ...metricsState,
            sensorsActive: resolvedActive,
            latencyMs: averagedLatency,
            lastCapture: resolvedCapture,
            timestamp: now
        };

        saveMetrics();
        updateMetricsUI();
    };

    const startMonitoring = () => {
        if (monitorInterval) {
            clearInterval(monitorInterval);
        }

        computeMetrics();
        monitorInterval = setInterval(computeMetrics, 15000);
    };

    loadStoredMetrics();
    updateMetricsUI();
    startMonitoring();

    if (form) {
        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            if (feedback) {
                feedback.textContent = '';
                feedback.classList.remove('error', 'success');
            }

            const timeValue = timeInput.value;
            const rotationValue = rotationInput.value;

            const autoFeedValue = autoFeedToggle ? autoFeedToggle.checked : true;

            if (!timeValue || !rotationValue) {
                if (feedback) {
                    feedback.textContent = 'Preencha horário e rotação para continuar.';
                    feedback.classList.add('error');
                }
                return;
            }

            const [hourString, minuteString] = timeValue.split(':');
            const payload = {
                hora: Number(hourString),
                minuto: Number(minuteString),
                rotacao: Number(rotationValue),
                auto_feed_enabled: autoFeedValue
            };

            try {
                const response = await fetch('/api/motor/schedule', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload)
                });

                const result = await response.json();

                if (!response.ok) {
                    throw new Error(result.error || 'Falha ao salvar agendamento.');
                }

                populateSchedule(result);
                if (feedback) {
                    feedback.textContent = 'Agendamento atualizado com sucesso!';
                    feedback.classList.add('success');
                }
            } catch (error) {
                console.error(error);
                if (feedback) {
                    feedback.textContent = error.message || 'Ocorreu um erro ao salvar.';
                    feedback.classList.add('error');
                }
            }
        });
    }

    if (runMotorButton) {
        runMotorButton.addEventListener('click', async () => {
            if (feedback) {
                feedback.textContent = '';
                feedback.classList.remove('error', 'success');
            }

            const rotationValue = rotationInput ? parseFloat(rotationInput.value) : NaN;

            if (!Number.isFinite(rotationValue) || rotationValue <= 0) {
                if (feedback) {
                    feedback.textContent = 'Informe uma rotação válida para girar o motor.';
                    feedback.classList.add('error');
                }
                return;
            }

            const originalLabel = runMotorButton.textContent;
            runMotorButton.disabled = true;
            runMotorButton.textContent = 'Girando...';

            try {
                const response = await fetch('/api/motor/run', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ rotacao: rotationValue })
                });

                const result = await response.json().catch(() => ({}));

                if (!response.ok) {
                    throw new Error(result.error || 'Falha ao acionar o motor.');
                }

                if (feedback) {
                    feedback.textContent = 'Motor acionado manualmente.';
                    feedback.classList.add('success');
                }
            } catch (error) {
                console.error(error);
                if (feedback) {
                    feedback.textContent = error.message || 'Não foi possível acionar o motor.';
                    feedback.classList.add('error');
                }
            } finally {
                runMotorButton.disabled = false;
                runMotorButton.textContent = originalLabel;
            }
        });
    }
});