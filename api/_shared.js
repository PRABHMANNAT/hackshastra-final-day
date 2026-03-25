import { Buffer } from 'node:buffer'

const MEDICAL_DEVELOPER_PROMPT = `You are SWASTH AI, a mental-health monitoring assistant operating inside a tactical clinical dashboard.
- Be calm, concise, and medically aware without overstating certainty.
- Use the live sensor snapshot and memory context when relevant.
- Focus on practical risk framing, grounding, next steps, and clinician-friendly summaries.
- Do not diagnose with certainty and do not invent measurements.
- If the user expresses imminent self-harm, suicidality, or danger, tell them to seek emergency help and a licensed crisis professional immediately.`

export async function readRawBody(request) {
  const chunks = []

  for await (const chunk of request) {
    chunks.push(chunk)
  }

  return Buffer.concat(chunks)
}

export async function readJsonBody(request) {
  const raw = await readRawBody(request)

  if (!raw.length) {
    return {}
  }

  try {
    return JSON.parse(raw.toString('utf8'))
  } catch {
    throw new Error('Invalid JSON payload received by the SWASTH API.')
  }
}

export function sendJson(response, statusCode, payload) {
  response.statusCode = statusCode
  response.setHeader('Content-Type', 'application/json')
  response.end(JSON.stringify(payload))
}

export async function parseUpstreamError(upstreamResponse) {
  const raw = await upstreamResponse.text()

  try {
    const parsed = raw ? JSON.parse(raw) : {}
    return parsed.error?.message || parsed.message || raw
  } catch {
    return raw
  }
}

export function extractStreamDelta(payload) {
  const content = payload.choices?.[0]?.delta?.content

  if (typeof content === 'string') {
    return content
  }

  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === 'string') {
          return part
        }

        return part?.text ?? ''
      })
      .join('')
  }

  return ''
}

export async function streamChatCompletion(response, env, body) {
  if (!env.VITE_OPENAI_API_KEY) {
    return sendJson(response, 500, {
      message: 'Missing VITE_OPENAI_API_KEY on the deployment.',
    })
  }

  const messages = Array.isArray(body.messages) ? body.messages : []
  const sensorSnapshot = String(body.sensorSnapshot || '').trim()
  const memoryContext = String(body.memoryContext || '').trim()

  const upstreamResponse = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${env.VITE_OPENAI_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: env.VITE_OPENAI_CHAT_MODEL || 'gpt-5.2-chat-latest',
      stream: true,
      temperature: 0.45,
      max_completion_tokens: 520,
      messages: [
        {
          role: 'developer',
          content: MEDICAL_DEVELOPER_PROMPT,
        },
        {
          role: 'developer',
          content: `Live sensor snapshot: ${sensorSnapshot || 'Unavailable'}\nMemory context: ${memoryContext || 'Unavailable'}`,
        },
        ...messages.map((message) => ({
          role: message.role === 'ai' ? 'assistant' : 'user',
          content: message.text,
        })),
      ],
    }),
  })

  if (!upstreamResponse.ok) {
    const message = await parseUpstreamError(upstreamResponse)
    return sendJson(response, upstreamResponse.status, {
      message: message || 'OpenAI chat stream failed.',
    })
  }

  if (!upstreamResponse.body) {
    return sendJson(response, 500, {
      message: 'OpenAI did not return a readable chat stream.',
    })
  }

  response.statusCode = 200
  response.setHeader('Content-Type', 'text/plain; charset=utf-8')
  response.setHeader('Cache-Control', 'no-cache')
  response.setHeader('Connection', 'keep-alive')
  response.setHeader('X-Accel-Buffering', 'no')
  response.flushHeaders?.()

  const decoder = new TextDecoder()
  const reader = upstreamResponse.body.getReader()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()

    if (done) {
      break
    }

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? ''

    for (const line of lines) {
      const trimmed = line.trim()

      if (!trimmed.startsWith('data:')) {
        continue
      }

      const payloadText = trimmed.slice(5).trim()

      if (!payloadText || payloadText === '[DONE]') {
        continue
      }

      try {
        const parsed = JSON.parse(payloadText)
        const delta = extractStreamDelta(parsed)

        if (delta) {
          response.write(delta)
        }
      } catch {
        // Ignore malformed stream fragments and continue the stream.
      }
    }
  }

  response.end()
}

export async function proxyTranscription(request, response, env) {
  if (!env.VITE_SPEECH_API_KEY) {
    return sendJson(response, 500, {
      message: 'Missing VITE_SPEECH_API_KEY on the deployment.',
    })
  }

  const rawBody = await readRawBody(request)
  const contentTypeHeader = Array.isArray(request.headers['content-type'])
    ? request.headers['content-type'][0]
    : request.headers['content-type']

  const upstreamResponse = await fetch('https://api.openai.com/v1/audio/transcriptions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${env.VITE_SPEECH_API_KEY}`,
      'Content-Type': contentTypeHeader || 'multipart/form-data',
    },
    body: rawBody,
  })

  const raw = await upstreamResponse.text()
  let parsed = {}

  try {
    parsed = raw ? JSON.parse(raw) : {}
  } catch {
    parsed = { text: '' }
  }

  if (!upstreamResponse.ok) {
    return sendJson(response, upstreamResponse.status, {
      message: parsed.error?.message || parsed.message || 'OpenAI transcription failed.',
    })
  }

  return sendJson(response, 200, {
    text: parsed.text || '',
  })
}

export async function proxySpeech(request, response, env) {
  if (!env.VITE_SPEECH_API_KEY) {
    return sendJson(response, 500, {
      message: 'Missing VITE_SPEECH_API_KEY on the deployment.',
    })
  }

  const body = await readJsonBody(request)
  const input = String(body.input || '').trim()

  if (!input) {
    return sendJson(response, 400, {
      message: 'Speech synthesis requires a non-empty input string.',
    })
  }

  const upstreamResponse = await fetch('https://api.openai.com/v1/audio/speech', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${env.VITE_SPEECH_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: env.VITE_SPEECH_MODEL || 'tts-1',
      voice: 'nova',
      input,
    }),
  })

  if (!upstreamResponse.ok) {
    const message = await parseUpstreamError(upstreamResponse)
    return sendJson(response, upstreamResponse.status, {
      message: message || 'OpenAI speech synthesis failed.',
    })
  }

  const audioBuffer = Buffer.from(await upstreamResponse.arrayBuffer())
  response.statusCode = 200
  response.setHeader('Content-Type', upstreamResponse.headers.get('content-type') || 'audio/mpeg')
  response.end(audioBuffer)
}
