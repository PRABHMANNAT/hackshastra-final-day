import process from 'node:process'
import { proxySpeech, sendJson } from './_shared.js'

export default async function handler(request, response) {
  if (request.method !== 'POST') {
    return sendJson(response, 405, {
      message: 'Method not allowed.',
    })
  }

  try {
    return await proxySpeech(request, response, process.env)
  } catch (error) {
    return sendJson(response, 500, {
      message: error.message || 'The speech function failed.',
    })
  }
}
