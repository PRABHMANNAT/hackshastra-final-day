import process from 'node:process'
import { streamChatCompletion, readJsonBody, sendJson } from './_shared.js'

export default async function handler(request, response) {
  if (request.method !== 'POST') {
    return sendJson(response, 405, {
      message: 'Method not allowed.',
    })
  }

  try {
    const body = await readJsonBody(request)

    return await streamChatCompletion(response, process.env, body)
  } catch (error) {
    return sendJson(response, 500, {
      message: error.message || 'The chat stream function failed.',
    })
  }
}
