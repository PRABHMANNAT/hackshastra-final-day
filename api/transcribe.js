import process from 'node:process'
import { proxyTranscription, sendJson } from './_shared.js'

export default async function handler(request, response) {
  if (request.method !== 'POST') {
    return sendJson(response, 405, {
      message: 'Method not allowed.',
    })
  }

  try {
    return await proxyTranscription(request, response, process.env)
  } catch (error) {
    return sendJson(response, 500, {
      message: error.message || 'The transcription function failed.',
    })
  }
}
