import { startTransition, useEffect, useEffectEvent, useRef, useState } from 'react'
import * as faceapi from '@vladmandic/face-api'
import '@tensorflow/tfjs-backend-webgl'
import {
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  Eye,
  LoaderCircle,
  Mic,
  SendHorizontal,
  Video,
  Zap,
} from 'lucide-react'
import { Area, AreaChart, ResponsiveContainer } from 'recharts'

const SENSOR_DEFINITIONS = [
  { key: 'heartRate', label: 'Heart Rate', unit: 'BPM', baseline: 68, floor: 60, ceiling: 85, precision: 0, drift: 2.6 },
  { key: 'spo2', label: 'SpO2', unit: '%', baseline: 96.8, floor: 95, ceiling: 99, precision: 1, drift: 0.45 },
  { key: 'skinConductance', label: 'Skin Conductance', unit: 'uS', baseline: 9.1, floor: 5, ceiling: 15, precision: 1, drift: 0.72 },
  { key: 'hrv', label: 'HRV', unit: 'ms', baseline: 68, floor: 50, ceiling: 100, precision: 0, drift: 3.4 },
]

const INITIAL_ALERTS = [
  { id: 'alert-1', severity: 'warning', message: 'Elevated stress markers detected', timestamp: '15:45:22' },
  { id: 'alert-2', severity: 'caution', message: 'Heart rate variability low', timestamp: '15:42:18' },
  { id: 'alert-3', severity: 'success', message: 'Positive mood indicators', timestamp: '15:40:05' },
]

const INITIAL_MESSAGES = [
  {
    id: 'message-1',
    role: 'ai',
    text: 'I have fused biometrics, facial affect, and memory context. Ask for a risk summary, recovery protocol, or session-ready briefing.',
    timestamp: '15:36:14',
  },
  { id: 'message-2', role: 'user', text: 'Summarize the patient state in one line.', timestamp: '15:36:43' },
  {
    id: 'message-3',
    role: 'ai',
    text: 'Stress is elevated, anxiety is moderate, and the patient remains coherent and coachable with targeted grounding support.',
    timestamp: '15:36:58',
  },
]

const ANOMALIES = [
  { label: 'Depression Risk', tone: 'danger', Icon: AlertTriangle },
  { label: 'ADHD Indicators', tone: 'warning', Icon: Zap },
  { label: 'High Stress', tone: 'pink', Icon: AlertCircle },
]

const FACIAL_BADGES = [
  { label: 'Facial Detection ✓', tone: 'success', Icon: CheckCircle2 },
  { label: 'Depression Markers', tone: 'warning', Icon: AlertTriangle },
  { label: 'Stress Level: High', tone: 'danger', Icon: AlertCircle },
  { label: 'Anxiety: Moderate', tone: 'caution', Icon: Zap },
  { label: 'Eye Strain Detected', tone: 'danger', Icon: Eye },
]

const MEMORY_TAGS = ['#anxiety', '#stress-management', '#cbt', '#grounding-exercises']

const MEMORY_CONTEXT = {
  summary: 'Patient history: 6 previous sessions.',
  narrative:
    'Diagnosed with generalized anxiety disorder. Responds well to CBT techniques. Prefers evening sessions and shows stronger recovery after guided grounding prompts.',
}

const LANDMARKS = [
  { x: 44, y: 36 },
  { x: 56, y: 36 },
  { x: 38, y: 46 },
  { x: 62, y: 46 },
  { x: 50, y: 54 },
  { x: 43, y: 65 },
  { x: 57, y: 65 },
  { x: 36, y: 29 },
  { x: 64, y: 29 },
]

const MIC_LABELS = {
  prompt: 'Permission required',
  granted: 'Mic ready',
  denied: 'Mic blocked',
  recording: 'Recording live',
  processing: 'Transcribing',
}

const AUDIO_NOTE_COPY =
  'Voice to text buffer is standing by. Click the microphone to capture speech, transcribe it with OpenAI, and route the reply back through the assistant.'

const OPENAI_TRANSCRIBE_MODEL = 'whisper-1'
const FACE_MODEL_PATH = '/models'
const FACE_SCAN_INTERVAL = 260
const FACE_DETECTOR_OPTIONS = new faceapi.TinyFaceDetectorOptions({
  inputSize: 224,
  scoreThreshold: 0.45,
})
const VIDEO_MIME_TYPES = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm']
const ARTBOARD_WIDTH = 1600
const ARTBOARD_HEIGHT = 900
const EXPRESSION_LABELS = {
  neutral: 'Neutral',
  happy: 'Happy',
  sad: 'Sad',
  angry: 'Angry',
  fearful: 'Fearful',
  disgusted: 'Disgusted',
  surprised: 'Surprised',
}

const timeFormatter = new Intl.DateTimeFormat('en-US', {
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
  hour12: false,
})

function createId(prefix = 'id') {
  return globalThis.crypto?.randomUUID?.() ?? `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max)
}

function randomBetween(min, max) {
  return Math.random() * (max - min) + min
}

function formatTimestamp(date = new Date()) {
  return timeFormatter.format(date)
}

function formatSensorValue(sensor, value) {
  return sensor.precision === 0 ? `${Math.round(value)}` : value.toFixed(sensor.precision)
}

function createSensorSeries(definition) {
  const points = []
  let current = definition.baseline + randomBetween(-definition.drift, definition.drift)

  for (let index = 0; index < 15; index += 1) {
    const drift = (definition.baseline - current) * 0.28
    const noise = randomBetween(-definition.drift, definition.drift)
    current = clamp(current + drift + noise, definition.floor, definition.ceiling)
    points.push(Number(current.toFixed(definition.precision)))
  }

  points.push(Number(definition.baseline.toFixed(definition.precision)))
  return points
}

function buildInitialSensors() {
  return SENSOR_DEFINITIONS.map((definition) => ({
    ...definition,
    series: createSensorSeries(definition),
  }))
}

function evolveSensor(sensor) {
  const current = sensor.series[sensor.series.length - 1]
  const driftToBaseline = (sensor.baseline - current) * 0.22
  const noise = randomBetween(-sensor.drift, sensor.drift)
  const next = clamp(current + driftToBaseline + noise, sensor.floor, sensor.ceiling)

  return {
    ...sensor,
    series: [...sensor.series.slice(-15), Number(next.toFixed(sensor.precision))],
  }
}

function createAlert(severity, message, timestamp = formatTimestamp()) {
  return { id: createId('alert'), severity, message, timestamp }
}

function createMessage(role, text, timestamp = formatTimestamp(), streaming = false) {
  return { id: createId('message'), role, text, timestamp, streaming }
}

function sensorStatusLine(sensors) {
  return sensors
    .map((sensor) => {
      const value = sensor.series[sensor.series.length - 1]
      return `${sensor.label}: ${formatSensorValue(sensor, value)} ${sensor.unit}`
    })
    .join(' | ')
}

function calculateMentalHealthScore(sensors) {
  const values = Object.fromEntries(
    sensors.map((sensor) => [sensor.key, sensor.series[sensor.series.length - 1]]),
  )

  const score =
    38.5 +
    Math.max(0, values.heartRate - 65) * 0.9 +
    Math.max(0, 97.5 - values.spo2) * 12 +
    Math.max(0, values.skinConductance - 7.5) * 3.8 +
    Math.max(0, 72 - values.hrv) * 0.55

  return clamp(Math.round(score), 28, 92)
}

function scoreDescriptor(score) {
  if (score >= 72) return 'Escalate review'
  if (score >= 55) return 'Watch closely'
  if (score >= 42) return 'Monitor trend'
  return 'Stable baseline'
}

function buildAdaptiveAlert(sensors) {
  const values = Object.fromEntries(
    sensors.map((sensor) => [sensor.key, sensor.series[sensor.series.length - 1]]),
  )

  if (values.skinConductance > 10.4 && values.hrv < 62) {
    return createAlert('danger', 'Rapid sympathetic activation detected')
  }

  if (values.heartRate > 74) {
    return createAlert('warning', 'Elevated stress markers detected')
  }

  if (values.hrv < 60) {
    return createAlert('caution', 'Heart rate variability low')
  }

  return createAlert('success', 'Positive mood indicators')
}

function normalizeMicPermission(state) {
  if (state === 'granted' || state === 'denied' || state === 'prompt') {
    return state
  }

  return 'prompt'
}

function formatPercent(value) {
  return `${Math.round(value * 100)}%`
}

function formatExpressionLabel(label) {
  return EXPRESSION_LABELS[label] || 'Scanning'
}

function pickVideoMimeType() {
  if (typeof MediaRecorder === 'undefined') {
    return ''
  }

  return VIDEO_MIME_TYPES.find((mimeType) => MediaRecorder.isTypeSupported(mimeType)) || ''
}

function distanceBetween(pointA, pointB) {
  return Math.hypot(pointA.x - pointB.x, pointA.y - pointB.y)
}

function calculateEyeAspectRatio(points) {
  if (!Array.isArray(points) || points.length < 6) {
    return null
  }

  const width = distanceBetween(points[0], points[3])

  if (!width) {
    return null
  }

  return (distanceBetween(points[1], points[5]) + distanceBetween(points[2], points[4])) / (2 * width)
}

function averageNumbers(values) {
  const validValues = values.filter((value) => typeof value === 'number' && Number.isFinite(value))

  if (!validValues.length) {
    return null
  }

  return validValues.reduce((sum, value) => sum + value, 0) / validValues.length
}

function describeInstantEyeState(averageEar) {
  if (typeof averageEar !== 'number' || Number.isNaN(averageEar)) {
    return {
      level: 'idle',
      tone: 'caution',
      label: 'Eye trace unavailable',
      detail: 'Landmarks are not stable enough yet',
    }
  }

  if (averageEar < 0.18) {
    return {
      level: 'high',
      tone: 'danger',
      label: 'Eye strain elevated',
      detail: 'Eyes appear heavily narrowed or closed',
    }
  }

  if (averageEar < 0.23) {
    return {
      level: 'warning',
      tone: 'warning',
      label: 'Possible eye strain',
      detail: 'Eye openness is trending low',
    }
  }

  return {
    level: 'stable',
    tone: 'success',
    label: 'Eye trace stable',
    detail: 'Eye openness is holding steady',
  }
}

function resolveEyeStrain(averageEar, tracker) {
  if (typeof averageEar !== 'number' || Number.isNaN(averageEar)) {
    tracker.smoothedEar = null
    tracker.lowSamples = 0
    tracker.closedSamples = 0
    return describeInstantEyeState(null)
  }

  tracker.smoothedEar = tracker.smoothedEar === null ? averageEar : tracker.smoothedEar * 0.7 + averageEar * 0.3

  const isLow = tracker.smoothedEar < 0.23
  const isClosed = tracker.smoothedEar < 0.18

  tracker.lowSamples = isLow ? Math.min(tracker.lowSamples + 1, 12) : Math.max(0, tracker.lowSamples - 1)
  tracker.closedSamples = isClosed ? Math.min(tracker.closedSamples + 1, 12) : Math.max(0, tracker.closedSamples - 1)

  if (tracker.closedSamples >= 3) {
    return {
      level: 'high',
      tone: 'danger',
      label: 'Eye strain elevated',
      detail: 'Eyes remain heavily narrowed across multiple frames',
    }
  }

  if (tracker.lowSamples >= 3) {
    return {
      level: 'warning',
      tone: 'warning',
      label: 'Possible eye strain',
      detail: 'Eye openness has stayed below the normal band',
    }
  }

  return describeInstantEyeState(tracker.smoothedEar)
}

function getDominantExpression(expressions = {}) {
  const [key = 'neutral', score = 0] =
    Object.entries(expressions).sort((entryA, entryB) => entryB[1] - entryA[1])[0] || []

  return {
    key,
    label: formatExpressionLabel(key),
    score,
  }
}

function buildEmptyCameraAnalysis() {
  return {
    faceCount: 0,
    fps: 0,
    dominantExpression: {
      key: 'neutral',
      label: 'Standby',
      score: 0,
    },
    eyeStrain: {
      level: 'idle',
      tone: 'caution',
      label: 'Eye trace unavailable',
      detail: 'Start the live camera to analyze facial markers',
    },
    faces: [],
    badges: [
      { label: 'Face tracking offline', tone: 'caution', Icon: AlertCircle },
      { label: 'Expression engine standing by', tone: 'warning', Icon: Zap },
      { label: 'Eye strain analysis idle', tone: 'caution', Icon: Eye },
    ],
  }
}

function buildCameraBadges({ faceCount, dominantExpression, eyeStrain, fps, isVideoRecording, modelStatus }) {
  const trackingBadge =
    faceCount > 0
      ? {
          label: `${faceCount} face${faceCount > 1 ? 's' : ''} locked`,
          tone: 'success',
          Icon: CheckCircle2,
        }
      : {
          label: modelStatus === 'loading' ? 'Loading face models' : 'Awaiting visible face',
          tone: modelStatus === 'loading' ? 'warning' : 'caution',
          Icon: modelStatus === 'loading' ? LoaderCircle : AlertCircle,
        }

  const expressionTone =
    dominantExpression.key === 'happy'
      ? 'success'
      : ['sad', 'angry', 'fearful', 'disgusted'].includes(dominantExpression.key)
        ? 'danger'
        : 'warning'

  return [
    trackingBadge,
    {
      label: faceCount > 0 ? `Expression: ${dominantExpression.label} ${formatPercent(dominantExpression.score)}` : 'Expression pending',
      tone: faceCount > 0 ? expressionTone : 'caution',
      Icon: faceCount > 0 ? Zap : AlertCircle,
    },
    {
      label: eyeStrain.label,
      tone: eyeStrain.tone,
      Icon: Eye,
    },
    {
      label: isVideoRecording ? 'Realtime clip recording' : 'Recorder idle',
      tone: isVideoRecording ? 'danger' : 'warning',
      Icon: Video,
    },
    {
      label: fps > 0 ? `Inference ${fps.toFixed(1)} FPS` : 'Inference waiting',
      tone: fps > 0 ? 'success' : 'caution',
      Icon: CheckCircle2,
    },
  ]
}

function createFaceSummary(detection) {
  const leftEar = calculateEyeAspectRatio(detection.landmarks?.getLeftEye?.() ?? [])
  const rightEar = calculateEyeAspectRatio(detection.landmarks?.getRightEye?.() ?? [])
  const eyeAspectRatio = averageNumbers([leftEar, rightEar])

  return {
    box: detection.detection.box,
    landmarks: detection.landmarks?.positions ?? [],
    dominantExpression: getDominantExpression(detection.expressions),
    eyeAspectRatio,
    eyeStrain: describeInstantEyeState(eyeAspectRatio),
  }
}

function createCameraSnapshot(detections, fps, tracker, isVideoRecording, modelStatus) {
  if (!detections.length) {
    tracker.smoothedEar = null
    tracker.lowSamples = 0
    tracker.closedSamples = 0

    const emptyState = buildEmptyCameraAnalysis()

    return {
      ...emptyState,
      fps,
      badges: buildCameraBadges({
        ...emptyState,
        fps,
        isVideoRecording,
        modelStatus,
      }),
    }
  }

  const mainDetection = detections.reduce((largest, current) => {
    const currentSize = current.detection.box.width * current.detection.box.height
    const largestSize = largest.detection.box.width * largest.detection.box.height
    return currentSize > largestSize ? current : largest
  }, detections[0])

  const mainLeftEar = calculateEyeAspectRatio(mainDetection.landmarks?.getLeftEye?.() ?? [])
  const mainRightEar = calculateEyeAspectRatio(mainDetection.landmarks?.getRightEye?.() ?? [])
  const smoothedEyeStrain = resolveEyeStrain(averageNumbers([mainLeftEar, mainRightEar]), tracker)
  const dominantExpression = getDominantExpression(mainDetection.expressions)
  const faces = detections.map((detection) => createFaceSummary(detection))

  return {
    faceCount: faces.length,
    fps,
    dominantExpression,
    eyeStrain: smoothedEyeStrain,
    faces,
    badges: buildCameraBadges({
      faceCount: faces.length,
      dominantExpression,
      eyeStrain: smoothedEyeStrain,
      fps,
      isVideoRecording,
      modelStatus,
    }),
  }
}

function drawCameraOverlay(canvas, analysis) {
  if (!canvas) {
    return
  }

  const context = canvas.getContext('2d')

  if (!context) {
    return
  }

  context.clearRect(0, 0, canvas.width, canvas.height)
  context.lineJoin = 'round'
  context.lineCap = 'round'
  context.textBaseline = 'top'
  context.font = '12px "DM Mono", monospace'

  analysis.faces.forEach((face) => {
    const expressionColor =
      face.dominantExpression.key === 'happy'
        ? '#38b859'
        : ['sad', 'angry', 'fearful', 'disgusted'].includes(face.dominantExpression.key)
          ? '#e24740'
          : '#eb9528'

    const { box } = face
    context.strokeStyle = expressionColor
    context.lineWidth = 2
    context.strokeRect(box.x, box.y, box.width, box.height)

    const labelLines = [
      `${face.dominantExpression.label} ${formatPercent(face.dominantExpression.score)}`,
      face.eyeStrain.label,
    ]
    const labelWidth = Math.max(...labelLines.map((line) => context.measureText(line).width)) + 18
    const labelX = clamp(box.x, 10, Math.max(10, canvas.width - labelWidth - 10))
    const labelY = box.y > 44 ? box.y - 38 : box.y + box.height + 10

    context.fillStyle = 'rgba(12, 12, 13, 0.82)'
    context.fillRect(labelX, labelY, labelWidth, 30)
    context.strokeStyle = expressionColor
    context.strokeRect(labelX, labelY, labelWidth, 30)

    context.fillStyle = '#f8f8f8'
    context.fillText(labelLines[0], labelX + 8, labelY + 5)

    context.fillStyle = face.eyeStrain.tone === 'danger' ? '#ff918b' : face.eyeStrain.tone === 'warning' ? '#f6c56b' : '#8ae2a0'
    context.fillText(labelLines[1], labelX + 8, labelY + 17)

    context.fillStyle = expressionColor
    face.landmarks.forEach((point) => {
      context.beginPath()
      context.arc(point.x, point.y, 1.5, 0, Math.PI * 2)
      context.fill()
    })
  })
}

function Sparkline({ sensor }) {
  const data = sensor.series.map((value, index) => ({ index, value }))

  return (
    <div className="sparkline">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data}>
          <defs>
            <linearGradient id={`spark-${sensor.key}`} x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="rgba(64,119,238,0.34)" />
              <stop offset="100%" stopColor="rgba(64,119,238,0.02)" />
            </linearGradient>
          </defs>
          <Area
            type="monotone"
            dataKey="value"
            stroke="#4077EE"
            strokeWidth={2}
            fill={`url(#spark-${sensor.key})`}
            dot={false}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

function SensorReadout({ sensors }) {
  return (
    <div className="rail-block rail-sensors">
      <div className="block-header">
        <span className="block-kicker">Left Rail</span>
        <h2 className="block-title">READINGS FROM SENSOR</h2>
      </div>

      <div className="sensor-stack">
        {sensors.map((sensor) => {
          const currentValue = sensor.series[sensor.series.length - 1]

          return (
            <article key={sensor.key} className="sensor-card">
              <div className="sensor-row">
                <span className="sensor-label">{sensor.label}</span>
                <div className="sensor-reading">
                  <span className="sensor-value">{formatSensorValue(sensor, currentValue)}</span>
                  <span className="sensor-unit">{sensor.unit}</span>
                </div>
              </div>
              <Sparkline sensor={sensor} />
            </article>
          )
        })}
      </div>
    </div>
  )
}

function AnomalyBlock() {
  return (
    <div className="rail-block rail-anomalies">
      <div className="block-header compact">
        <span className="block-kicker">Display if</span>
        <h2 className="block-title">ANOMALY DETECTED</h2>
      </div>

      <div className="badge-row">
        {ANOMALIES.map((anomaly) => {
          const IconComponent = anomaly.Icon
          return (
            <span key={anomaly.label} className={`signal-badge badge-${anomaly.tone}`}>
              <IconComponent size={14} strokeWidth={2} />
              {anomaly.label}
            </span>
          )
        })}
      </div>
    </div>
  )
}

function AlertsBlock({ alerts }) {
  return (
    <div className="rail-block rail-alerts">
      <div className="block-header compact">
        <span className="block-kicker">Bottom Left</span>
        <h2 className="block-title">REAL TIME ALERTS</h2>
      </div>

      <div className="alerts-list">
        {alerts.map((alert) => (
          <article key={alert.id} className={`alert-card severity-${alert.severity}`}>
            <div className="alert-head">
              <span className="alert-time">{alert.timestamp}</span>
            </div>
            <p className="alert-text">{alert.message}</p>
          </article>
        ))}
      </div>
    </div>
  )
}

function MessagePreview({ message }) {
  const isAi = message.role === 'ai'

  return (
    <article className={`preview-bubble ${isAi ? 'preview-ai' : 'preview-user'}`}>
      <div className="preview-head">
        <span className={`preview-role ${isAi ? 'role-ai' : 'role-user'}`}>{isAi ? 'SWASTH AI' : 'USER'}</span>
        <span className="preview-time">{message.timestamp}</span>
      </div>
      <p className="preview-text">{message.text || (message.streaming ? 'Streaming response...' : '')}</p>
    </article>
  )
}

function CameraBlock({
  cameraAnalysis,
  cameraError,
  cameraModelStatus,
  cameraState,
  isVideoRecording,
  lastVideoClipUrl,
  onToggleRecording,
  overlayRef,
  videoRef,
}) {
  return (
    <div className="rail-block rail-camera">
      <div className="block-header">
        <span className="block-kicker">Top Right</span>
        <h2 className="block-title">LIVE CAMERA RECORDING</h2>
      </div>

      <div className={`camera-box ${cameraState === 'live' ? 'camera-box-live' : ''}`}>
        <div className="camera-bar">
          <span className="camera-tag">{isVideoRecording ? 'REC' : cameraModelStatus === 'loading' ? 'LOAD' : 'LIVE'}</span>
          <span className={`camera-dot ${isVideoRecording ? 'active' : ''}`} />
          <span>{cameraAnalysis.faceCount ? `${cameraAnalysis.faceCount} FACE` : cameraState === 'booting' ? 'BOOT' : 'SCAN'}</span>
          <span>{cameraAnalysis.fps ? `${cameraAnalysis.fps.toFixed(1)} FPS` : '-- FPS'}</span>
        </div>
        <span className="camera-bracket bracket-tl" aria-hidden="true" />
        <span className="camera-bracket bracket-tr" aria-hidden="true" />
        <span className="camera-bracket bracket-bl" aria-hidden="true" />
        <span className="camera-bracket bracket-br" aria-hidden="true" />

        <video ref={videoRef} className="camera-video" playsInline muted autoPlay />
        <canvas ref={overlayRef} className="camera-overlay" />

        {cameraState !== 'live' ? (
          <div className="camera-silhouette">
            <div className="silhouette-core" />
            {LANDMARKS.map((landmark, index) => (
              <span key={index} className="landmark" style={{ left: `${landmark.x}%`, top: `${landmark.y}%` }} />
            ))}
          </div>
        ) : null}

        <div className="camera-footer">
          <span className={`camera-pill badge-${cameraAnalysis.eyeStrain.tone}`}>{cameraAnalysis.eyeStrain.label}</span>
          <span className="camera-pill">
            {cameraAnalysis.faceCount
              ? `${cameraAnalysis.dominantExpression.label} ${formatPercent(cameraAnalysis.dominantExpression.score)}`
              : cameraModelStatus === 'loading'
                ? 'Models loading'
                : 'Awaiting face'}
          </span>
        </div>

        {cameraError ? <div className="camera-error">{cameraError}</div> : null}
      </div>

      <button type="button" className="record-button" onClick={onToggleRecording}>
        <Video size={18} strokeWidth={2} />
        <span>{isVideoRecording ? 'Stop Recording' : 'Start Live Recording'}</span>
      </button>

      {lastVideoClipUrl ? (
        <a className="camera-download" href={lastVideoClipUrl} download="swasth-session.webm">
          Download last clip
        </a>
      ) : null}
    </div>
  )
}

function AnalysisBlock({ analysis, score }) {
  return (
    <div className="rail-block rail-analysis">
      <div className="block-header compact">
        <span className="block-kicker">Middle Right</span>
        <h2 className="block-title">FACIAL ANALYSIS</h2>
      </div>

      <div className="analysis-grid">
        <div className="analysis-badges">
          {analysis.badges.map((badge) => {
            const IconComponent = badge.Icon

            return (
              <span key={badge.label} className={`analysis-badge badge-${badge.tone}`}>
                <IconComponent size={13} strokeWidth={2} />
                {badge.label}
              </span>
            )
          })}
        </div>

        <div className="analysis-stats">
          <article className="analysis-stat">
            <span className="analysis-stat-label">Faces</span>
            <strong className="analysis-stat-value">{analysis.faceCount || 0}</strong>
          </article>
          <article className="analysis-stat">
            <span className="analysis-stat-label">Affect</span>
            <strong className="analysis-stat-value">{analysis.dominantExpression.label}</strong>
          </article>
          <article className="analysis-stat">
            <span className="analysis-stat-label">Eye Load</span>
            <strong className="analysis-stat-value">{analysis.eyeStrain.label}</strong>
          </article>
          <article className="analysis-stat">
            <span className="analysis-stat-label">Inference</span>
            <strong className="analysis-stat-value">{analysis.fps ? `${analysis.fps.toFixed(1)} FPS` : 'Standby'}</strong>
          </article>
        </div>

        <div className="score-panel">
          <div className="score-head">
            <span className="score-label">MULTIPLE FACTOR</span>
            <span className="score-value">{score}</span>
          </div>
          <div className="score-track">
            <div className="score-fill" style={{ width: `${score}%` }} />
          </div>
          <div className="score-foot">
            <span>0</span>
            <span>{scoreDescriptor(score)}</span>
            <span>100</span>
          </div>
        </div>
      </div>
    </div>
  )
}

function MemoryBlock() {
  return (
    <div className="rail-block rail-memory">
      <div className="block-header compact">
        <span className="block-kicker">Bottom Right</span>
        <h2 className="block-title">MEMORY CONTEXT</h2>
      </div>

      <div className="memory-box">
        <p className="memory-summary">{MEMORY_CONTEXT.summary}</p>
        <p className="memory-text">{MEMORY_CONTEXT.narrative}</p>
        <div className="memory-tags">
          {MEMORY_TAGS.map((tag) => (
            <span key={tag} className="memory-tag">
              {tag}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}

function App() {
  const [sensors, setSensors] = useState(() => buildInitialSensors())
  const [alerts, setAlerts] = useState(INITIAL_ALERTS)
  const [chatMessages, setChatMessages] = useState(INITIAL_MESSAGES)
  const [draft, setDraft] = useState('')
  const [chatError, setChatError] = useState('')
  const [isChatStreaming, setIsChatStreaming] = useState(false)
  const [micStatus, setMicStatus] = useState('prompt')
  const [voiceBufferText, setVoiceBufferText] = useState(AUDIO_NOTE_COPY)
  const [isRecording, setIsRecording] = useState(false)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [isVideoRecording, setIsVideoRecording] = useState(false)
  const [cameraState, setCameraState] = useState('idle')
  const [cameraError, setCameraError] = useState('')
  const [cameraModelStatus, setCameraModelStatus] = useState('idle')
  const [cameraAnalysis, setCameraAnalysis] = useState(() => buildEmptyCameraAnalysis())
  const [lastVideoClipUrl, setLastVideoClipUrl] = useState('')
  const [isCompactViewport, setIsCompactViewport] = useState(false)
  const [viewportScale, setViewportScale] = useState(1)

  const mediaRecorderRef = useRef(null)
  const mediaStreamRef = useRef(null)
  const recordingTimeoutRef = useRef(null)
  const audioChunksRef = useRef([])
  const audioPlayerRef = useRef(null)
  const audioUrlRef = useRef(null)
  const chatMessagesRef = useRef(chatMessages)
  const sensorsRef = useRef(sensors)
  const videoElementRef = useRef(null)
  const overlayCanvasRef = useRef(null)
  const cameraStreamRef = useRef(null)
  const videoRecorderRef = useRef(null)
  const cameraChunksRef = useRef([])
  const cameraLoopTimeoutRef = useRef(null)
  const isCameraInferenceRunningRef = useRef(false)
  const faceModelsReadyRef = useRef(false)
  const lastVideoClipRef = useRef(null)
  const eyeStrainTrackerRef = useRef({
    smoothedEar: null,
    lowSamples: 0,
    closedSamples: 0,
  })
  const lastCameraAlertRef = useRef({
    key: '',
    timestamp: 0,
  })

  const score = calculateMentalHealthScore(sensors)
  const statusLine = sensorStatusLine(sensors)
  const recentMessages = chatMessages.slice(-2)
  const latestAiMessage = [...chatMessages].reverse().find((message) => message.role === 'ai')
  const cameraStatusLabel = isVideoRecording
    ? 'Camera recording live'
    : cameraModelStatus === 'loading'
      ? 'Camera models loading'
      : 'Camera standby'

  useEffect(() => {
    chatMessagesRef.current = chatMessages
  }, [chatMessages])

  useEffect(() => {
    sensorsRef.current = sensors
  }, [sensors])

  const tickSensors = useEffectEvent(() => {
    startTransition(() => {
      setSensors((currentSensors) => currentSensors.map((sensor) => evolveSensor(sensor)))
    })
  })

  const appendAlert = useEffectEvent(() => {
    setAlerts((currentAlerts) => [buildAdaptiveAlert(sensorsRef.current), ...currentAlerts].slice(0, 8))
  })

  useEffect(() => {
    const interval = window.setInterval(() => {
      tickSensors()
    }, 1700)

    return () => window.clearInterval(interval)
  }, [])

  useEffect(() => {
    const interval = window.setInterval(() => {
      appendAlert()
    }, 11000)

    return () => window.clearInterval(interval)
  }, [])

  function resetCameraCanvas() {
    const canvas = overlayCanvasRef.current

    if (!canvas) {
      return
    }

    const context = canvas.getContext('2d')

    if (!context) {
      return
    }

    context.clearRect(0, 0, canvas.width, canvas.height)
  }

  function detachCameraStream() {
    window.clearTimeout(cameraLoopTimeoutRef.current)
    isCameraInferenceRunningRef.current = false

    const videoElement = videoElementRef.current

    if (videoElement) {
      videoElement.pause()
      videoElement.srcObject = null
    }

    cameraStreamRef.current?.getTracks().forEach((track) => track.stop())
    cameraStreamRef.current = null
    resetCameraCanvas()
  }

  useEffect(() => {
    let permissionState

    async function checkMicPermission() {
      if (!navigator.permissions?.query) {
        return
      }

      try {
        permissionState = await navigator.permissions.query({ name: 'microphone' })
        setMicStatus(normalizeMicPermission(permissionState.state))
        permissionState.onchange = () => {
          setMicStatus(normalizeMicPermission(permissionState.state))
        }
      } catch {
        // Permissions API is optional across browsers.
      }
    }

    checkMicPermission()

    return () => {
      if (permissionState) {
        permissionState.onchange = null
      }

      window.clearTimeout(recordingTimeoutRef.current)
      window.clearTimeout(cameraLoopTimeoutRef.current)
      mediaRecorderRef.current?.stop()
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop())
      videoRecorderRef.current?.stop()
      cameraStreamRef.current?.getTracks().forEach((track) => track.stop())
      audioPlayerRef.current?.pause()

      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current)
      }

      if (lastVideoClipRef.current) {
        URL.revokeObjectURL(lastVideoClipRef.current)
      }
    }
  }, [])

  useEffect(() => {
    function updateViewportScale() {
      const compactViewport = window.innerWidth <= 980
      setIsCompactViewport(compactViewport)

      if (compactViewport) {
        setViewportScale(1)
        return
      }

      const padding = 24
      const widthRatio = (window.innerWidth - padding) / ARTBOARD_WIDTH
      const heightRatio = (window.innerHeight - padding) / ARTBOARD_HEIGHT
      const nextScale = Math.min(1, widthRatio, heightRatio)
      setViewportScale(Number.isFinite(nextScale) ? Math.max(nextScale, 0.58) : 1)
    }

    updateViewportScale()
    window.addEventListener('resize', updateViewportScale)

    return () => window.removeEventListener('resize', updateViewportScale)
  }, [])

  function updateStreamingMessage(messageId, nextText, isComplete = false) {
    setChatMessages((currentMessages) =>
      currentMessages.map((message) =>
        message.id === messageId ? { ...message, text: nextText, streaming: !isComplete } : message,
      ),
    )
  }

  async function playSpeech(text) {
    try {
      const response = await fetch('/api/speech', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ input: text }),
      })

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}))
        throw new Error(payload.message || 'Speech synthesis failed.')
      }

      const audioBlob = await response.blob()

      if (audioPlayerRef.current) {
        audioPlayerRef.current.pause()
      }

      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current)
      }

      const audioUrl = URL.createObjectURL(audioBlob)
      const player = new Audio(audioUrl)
      audioPlayerRef.current = player
      audioUrlRef.current = audioUrl
      player.onended = () => {
        if (audioUrlRef.current) {
          URL.revokeObjectURL(audioUrlRef.current)
          audioUrlRef.current = null
        }
      }

      await player.play().catch(() => {})
    } catch (error) {
      setVoiceBufferText(error.message)
    }
  }

  async function streamAssistantReply(messageHistory, { speakReply = false } = {}) {
    const assistantMessage = createMessage('ai', '', formatTimestamp(), true)
    setChatMessages((currentMessages) => [...currentMessages, assistantMessage])
    setChatError('')
    setIsChatStreaming(true)

    try {
      const response = await fetch('/api/chat-stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: messageHistory,
          sensorSnapshot: sensorStatusLine(sensorsRef.current),
          memoryContext: `${MEMORY_CONTEXT.summary} ${MEMORY_CONTEXT.narrative}`,
        }),
      })

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}))
        throw new Error(payload.message || 'OpenAI chat stream failed.')
      }

      if (!response.body) {
        throw new Error('The browser did not receive a readable chat stream.')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let nextText = ''

      while (true) {
        const { value, done } = await reader.read()

        if (done) {
          break
        }

        nextText += decoder.decode(value, { stream: true })
        updateStreamingMessage(assistantMessage.id, nextText)
      }

      nextText += decoder.decode()

      if (!nextText.trim()) {
        throw new Error('OpenAI returned an empty medical response.')
      }

      updateStreamingMessage(assistantMessage.id, nextText.trim(), true)

      if (speakReply) {
        await playSpeech(nextText.trim())
      }
    } catch (error) {
      updateStreamingMessage(
        assistantMessage.id,
        'The assistant stream was interrupted. Re-run the request to continue.',
        true,
      )
      setChatError(error.message)
    } finally {
      setIsChatStreaming(false)
    }
  }

  async function handleSubmit(event) {
    event.preventDefault()
    const nextPrompt = draft.trim()

    if (!nextPrompt || isChatStreaming || isTranscribing) {
      return
    }

    const userMessage = createMessage('user', nextPrompt)
    const nextHistory = [...chatMessagesRef.current, userMessage]

    setDraft('')
    setChatMessages(nextHistory)
    await streamAssistantReply(nextHistory)
  }

  async function transcribeAudio(audioBlob) {
    const formData = new FormData()
    formData.append('file', audioBlob, `voice-note-${Date.now()}.webm`)
    formData.append('model', OPENAI_TRANSCRIBE_MODEL)

    const response = await fetch('/api/transcribe', {
      method: 'POST',
      body: formData,
    })

    const payload = await response.json().catch(() => ({}))

    if (!response.ok) {
      throw new Error(payload.message || 'OpenAI transcription failed.')
    }

    if (!payload.text?.trim()) {
      throw new Error('No transcript returned from the audio note.')
    }

    return payload.text.trim()
  }

  async function ensureFaceModelsReady() {
    if (faceModelsReadyRef.current) {
      return
    }

    setCameraModelStatus('loading')

    try {
      if (faceapi.tf.getBackend() !== 'webgl') {
        await faceapi.tf.setBackend('webgl')
      }

      await faceapi.tf.ready()

      if (faceapi.tf?.env()?.flagRegistry?.CANVAS2D_WILL_READ_FREQUENTLY) {
        faceapi.tf.env().set('CANVAS2D_WILL_READ_FREQUENTLY', true)
      }

      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(FACE_MODEL_PATH),
        faceapi.nets.faceLandmark68TinyNet.loadFromUri(FACE_MODEL_PATH),
        faceapi.nets.faceExpressionNet.loadFromUri(FACE_MODEL_PATH),
      ])

      faceModelsReadyRef.current = true
      setCameraModelStatus('ready')
    } catch (error) {
      setCameraModelStatus('error')
      throw new Error(error.message || 'Face analysis models could not be loaded.')
    }
  }

  function finalizeVideoClip(recorder) {
    const clipBlob = new Blob(cameraChunksRef.current, {
      type: recorder.mimeType || 'video/webm',
    })

    cameraChunksRef.current = []

    if (!clipBlob.size) {
      return
    }

    if (lastVideoClipRef.current) {
      URL.revokeObjectURL(lastVideoClipRef.current)
    }

    const clipUrl = URL.createObjectURL(clipBlob)
    lastVideoClipRef.current = clipUrl
    setLastVideoClipUrl(clipUrl)
  }

  async function waitForVideoReady(videoElement) {
    if (videoElement.readyState >= 2) {
      return
    }

    await new Promise((resolve) => {
      const handleLoadedData = () => {
        videoElement.removeEventListener('loadeddata', handleLoadedData)
        resolve()
      }

      videoElement.addEventListener('loadeddata', handleLoadedData)
    })
  }

  function stopRecording() {
    window.clearTimeout(recordingTimeoutRef.current)

    const recorder = mediaRecorderRef.current
    if (recorder && recorder.state !== 'inactive') {
      recorder.stop()
    }
  }

  async function startRecording() {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setMicStatus('denied')
      setVoiceBufferText('This browser cannot capture microphone audio. Use a Chromium-based browser or Safari.')
      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)

      mediaStreamRef.current = stream
      mediaRecorderRef.current = recorder
      audioChunksRef.current = []

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      recorder.onstop = async () => {
        mediaStreamRef.current?.getTracks().forEach((track) => track.stop())
        mediaStreamRef.current = null

        const audioBlob = new Blob(audioChunksRef.current, {
          type: recorder.mimeType || 'audio/webm',
        })

        setIsRecording(false)

        if (!audioBlob.size) {
          setMicStatus('granted')
          setIsTranscribing(false)
          setVoiceBufferText('No audio was captured. Try again with the microphone closer.')
          return
        }

        try {
          setMicStatus('processing')
          setIsTranscribing(true)
          const transcript = await transcribeAudio(audioBlob)
          setVoiceBufferText(transcript)

          const userMessage = createMessage('user', transcript)
          const nextHistory = [...chatMessagesRef.current, userMessage]
          setChatMessages(nextHistory)
          await streamAssistantReply(nextHistory, { speakReply: true })
          setMicStatus('granted')
        } catch (error) {
          setMicStatus('granted')
          setChatError(error.message)
          setVoiceBufferText(error.message)
        } finally {
          setIsTranscribing(false)
        }
      }

      recorder.start()
      setIsRecording(true)
      setMicStatus('recording')
      setVoiceBufferText('Voice to text recording live. Click again to stop and hand the audio to OpenAI.')

      recordingTimeoutRef.current = window.setTimeout(() => {
        stopRecording()
      }, 5800)
    } catch {
      setMicStatus('denied')
      setVoiceBufferText('Microphone permission was denied. Allow access to use voice capture.')
    }
  }

  async function handleMicClick() {
    if (isRecording) {
      stopRecording()
      return
    }

    if (isChatStreaming || isTranscribing) {
      return
    }

    await startRecording()
  }

  const runCameraInference = useEffectEvent(async () => {
    const videoElement = videoElementRef.current
    const canvas = overlayCanvasRef.current

    if (!videoElement || !canvas || !isVideoRecording || videoElement.readyState < 2) {
      return
    }

    if (isCameraInferenceRunningRef.current) {
      return
    }

    isCameraInferenceRunningRef.current = true
    const frameStart = performance.now()

    try {
      canvas.width = videoElement.videoWidth
      canvas.height = videoElement.videoHeight

      const detections = await faceapi
        .detectAllFaces(videoElement, FACE_DETECTOR_OPTIONS)
        .withFaceLandmarks(true)
        .withFaceExpressions()

      const fps = 1000 / Math.max(performance.now() - frameStart, 1)
      const nextAnalysis = createCameraSnapshot(
        detections,
        fps,
        eyeStrainTrackerRef.current,
        isVideoRecording,
        cameraModelStatus,
      )

      drawCameraOverlay(canvas, nextAnalysis)
      startTransition(() => {
        setCameraAnalysis(nextAnalysis)
      })
    } catch (error) {
      window.clearTimeout(cameraLoopTimeoutRef.current)
      isCameraInferenceRunningRef.current = false
      detachCameraStream()
      setIsVideoRecording(false)
      setCameraState('error')
      setCameraAnalysis(buildEmptyCameraAnalysis())
      setCameraError(error.message || 'Camera inference failed during live analysis.')
      return
    }

    isCameraInferenceRunningRef.current = false

    if (isVideoRecording) {
      cameraLoopTimeoutRef.current = window.setTimeout(() => {
        runCameraInference()
      }, FACE_SCAN_INTERVAL)
    }
  })

  useEffect(() => {
    if (!isVideoRecording) {
      return
    }

    runCameraInference()

    return () => {
      window.clearTimeout(cameraLoopTimeoutRef.current)
      isCameraInferenceRunningRef.current = false
    }
  }, [cameraModelStatus, isVideoRecording])

  useEffect(() => {
    if (!isVideoRecording || !cameraAnalysis.faceCount) {
      return
    }

    const highRiskExpression =
      ['sad', 'angry', 'fearful'].includes(cameraAnalysis.dominantExpression.key) &&
      cameraAnalysis.dominantExpression.score >= 0.68
    const highEyeStrain = cameraAnalysis.eyeStrain.level === 'high'
    const now = Date.now()

    if (
      highEyeStrain &&
      (lastCameraAlertRef.current.key !== 'eye' || now - lastCameraAlertRef.current.timestamp > 12000)
    ) {
      setAlerts((currentAlerts) =>
        [createAlert('danger', 'Realtime camera analysis suggests elevated eye strain.'), ...currentAlerts].slice(0, 8),
      )
      lastCameraAlertRef.current = { key: 'eye', timestamp: now }
      return
    }

    if (
      highRiskExpression &&
      (lastCameraAlertRef.current.key !== cameraAnalysis.dominantExpression.key ||
        now - lastCameraAlertRef.current.timestamp > 12000)
    ) {
      setAlerts((currentAlerts) =>
        [
          createAlert(
            'warning',
            `Facial affect trend leaning ${cameraAnalysis.dominantExpression.label.toLowerCase()} at ${formatPercent(cameraAnalysis.dominantExpression.score)} confidence.`,
          ),
          ...currentAlerts,
        ].slice(0, 8),
      )
      lastCameraAlertRef.current = {
        key: cameraAnalysis.dominantExpression.key,
        timestamp: now,
      }
    }
  }, [cameraAnalysis, isVideoRecording])

  async function startVideoRecording() {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setCameraState('error')
      setCameraError('This browser cannot access or record live camera video.')
      return
    }

    setCameraError('')
    setCameraState('booting')
    setCameraAnalysis(buildEmptyCameraAnalysis())

    try {
      await ensureFaceModelsReady()

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: 'user',
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      })

      const videoElement = videoElementRef.current

      if (!videoElement) {
        throw new Error('Live camera surface could not attach to the video element.')
      }

      cameraStreamRef.current = stream
      videoElement.srcObject = stream
      videoElement.muted = true

      await waitForVideoReady(videoElement)
      await videoElement.play()

      const mimeType = pickVideoMimeType()
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
      cameraChunksRef.current = []
      videoRecorderRef.current = recorder

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          cameraChunksRef.current.push(event.data)
        }
      }

      recorder.onstop = () => {
        finalizeVideoClip(recorder)
        videoRecorderRef.current = null
      }

      recorder.start(1000)
      setIsVideoRecording(true)
      setCameraState('live')
      setAlerts((currentAlerts) =>
        [createAlert('warning', 'Live camera recording engaged for clinician review.'), ...currentAlerts].slice(0, 8),
      )
    } catch (error) {
      detachCameraStream()
      setIsVideoRecording(false)
      setCameraState('error')
      setCameraError(error.message || 'Camera access failed.')
    }
  }

  function stopVideoRecording() {
    window.clearTimeout(cameraLoopTimeoutRef.current)
    isCameraInferenceRunningRef.current = false

    const recorder = videoRecorderRef.current

    if (recorder && recorder.state !== 'inactive') {
      recorder.stop()
    }

    detachCameraStream()
    eyeStrainTrackerRef.current = {
      smoothedEar: null,
      lowSamples: 0,
      closedSamples: 0,
    }
    setIsVideoRecording(false)
    setCameraState('idle')
    setCameraAnalysis(buildEmptyCameraAnalysis())
    setAlerts((currentAlerts) =>
      [createAlert('success', 'Camera clip archived to the monitoring session.'), ...currentAlerts].slice(0, 8),
    )
  }

  async function handleVideoRecordingToggle() {
    if (isVideoRecording) {
      stopVideoRecording()
      return
    }

    await startVideoRecording()
  }

  return (
    <div className="viewport-shell">
      <div
        className="fit-frame"
        style={{
          width: isCompactViewport ? '100%' : `${ARTBOARD_WIDTH * viewportScale}px`,
          height: isCompactViewport ? 'auto' : `${ARTBOARD_HEIGHT * viewportScale}px`,
        }}
      >
        <div
          className="interface-shell"
          style={{
            width: isCompactViewport ? '100%' : `${ARTBOARD_WIDTH}px`,
            height: isCompactViewport ? 'auto' : `${ARTBOARD_HEIGHT}px`,
            minHeight: isCompactViewport ? 'calc(100svh - 24px)' : `${ARTBOARD_HEIGHT}px`,
            transform: isCompactViewport ? 'none' : `scale(${viewportScale})`,
          }}
        >
          <div className="orb orb-left" aria-hidden="true" />
          <div className="orb orb-center" aria-hidden="true" />
          <div className="orb orb-right" aria-hidden="true" />
          <div className="grain" aria-hidden="true" />
          <div className="scanlines" aria-hidden="true" />

          <header className="surface-header">
            <div className="surface-brand">
              <span className="surface-kicker">TACTICAL OS</span>
              <h1 className="surface-title">SWASTH AI</h1>
            </div>
            <div className="surface-meta">
              <span className="surface-pill">SWASTH AI CLINICAL SURFACE</span>
              <span className="surface-score">
                {score}
                <span>{scoreDescriptor(score)}</span>
              </span>
            </div>
          </header>

          <div className="status-ribbon">
            <span>{statusLine}</span>
            <span>
              {MIC_LABELS[micStatus]} / {cameraStatusLabel}
            </span>
          </div>

          <main className="blueprint">
            <aside className="left-rail">
              <SensorReadout sensors={sensors} />
              <AnomalyBlock />
              <AlertsBlock alerts={alerts} />
            </aside>

            <section className="center-stage">
              <div className="stage-topline">
                <span className="stage-kicker">Speech Model / OpenAI Creative</span>
                <button
                  type="button"
                  className={`stage-mic ${isRecording ? 'recording' : ''}`}
                  onClick={handleMicClick}
                  disabled={isChatStreaming || isTranscribing}
                >
                  {isRecording || isTranscribing ? (
                    <LoaderCircle size={16} strokeWidth={2} className="spin" />
                  ) : (
                    <Mic size={16} strokeWidth={2} />
                  )}
                  {MIC_LABELS[micStatus]}
                </button>
              </div>

              <div className={`assistant-orb-shell state-${micStatus}`}>
                <button
                  type="button"
                  className="assistant-orb"
                  onClick={handleMicClick}
                  disabled={isChatStreaming || isTranscribing}
                  aria-label={isRecording ? 'Stop voice capture' : 'Start voice capture'}
                >
                  <span className="assistant-core" />
                  <span className="assistant-ring ring-a" />
                  <span className="assistant-ring ring-b" />
                  <span className="assistant-ring ring-c" />
                </button>
              </div>

              <div className="type-animation">
                <span className="stage-kicker">TYPE ANIMATION</span>
                <p className={`type-copy ${latestAiMessage?.streaming ? 'streaming' : ''}`}>
                  {latestAiMessage?.text || AUDIO_NOTE_COPY}
                </p>
              </div>

              <div className="preview-strip">
                {recentMessages.map((message) => (
                  <MessagePreview key={message.id} message={message} />
                ))}
              </div>

              <div className="voice-buffer">VOICE TO TEXT / {voiceBufferText}</div>

              <form className="chat-dock" onSubmit={handleSubmit}>
                <button
                  type="button"
                  className={`dock-button ${isRecording ? 'recording' : ''}`}
                  onClick={handleMicClick}
                  disabled={isChatStreaming || isTranscribing}
                  aria-label={isRecording ? 'Stop microphone recording' : 'Start microphone recording'}
                >
                  {isRecording || isTranscribing ? (
                    <LoaderCircle size={18} strokeWidth={2} className="spin" />
                  ) : (
                    <Mic size={18} strokeWidth={2} />
                  )}
                </button>
                <input
                  className="chat-input"
                  type="text"
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  placeholder="Chatbot / Ask your AI health assistant..."
                  disabled={isChatStreaming || isTranscribing}
                />
                <button
                  type="submit"
                  className="dock-button send"
                  disabled={!draft.trim() || isChatStreaming || isTranscribing}
                  aria-label="Send chat message"
                >
                  <SendHorizontal size={18} strokeWidth={2} />
                </button>
              </form>

              {chatError ? <p className="dock-error">{chatError}</p> : null}
            </section>

            <aside className="right-rail">
              <CameraBlock
                cameraAnalysis={cameraAnalysis}
                cameraError={cameraError}
                cameraModelStatus={cameraModelStatus}
                cameraState={cameraState}
                isVideoRecording={isVideoRecording}
                lastVideoClipUrl={lastVideoClipUrl}
                onToggleRecording={handleVideoRecordingToggle}
                overlayRef={overlayCanvasRef}
                videoRef={videoElementRef}
              />
              <AnalysisBlock analysis={cameraAnalysis} score={score} />
              <MemoryBlock />
            </aside>
          </main>
        </div>
      </div>
    </div>
  )
}

export default App
