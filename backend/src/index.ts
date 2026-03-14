import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { SarvamAIClient } from 'sarvamai'
import { GoogleGenAI } from '@google/genai'

const sarvam = new SarvamAIClient({
  apiSubscriptionKey: process.env.SARVAM_API_KEY,
})

const genai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY })

const app = new Hono()

app.use('*', cors())

app.get('/', (c) => c.text('Voice Agent API'))

// POST /api/chat
// FormData: audio (File), history (JSON string), system_prompt (string), language_code (string)
// Returns: { transcription, response, audio }
app.post('/api/chat', async (c) => {
  const formData = await c.req.formData()
  const audioFile = formData.get('audio') as File | null
  const historyJson = (formData.get('history') as string) || '[]'
  const systemPrompt =
    (formData.get('system_prompt') as string) || 'You are a helpful assistant.'
  const languageCode = (formData.get('language_code') as string) || 'en-IN'

  if (!audioFile) {
    return c.json({ error: 'No audio file provided' }, 400)
  }

  // STT: transcribe audio
  const sttResponse = await sarvam.speechToText.transcribe({
    file: audioFile,
    language_code: 'unknown' as any,
  })

  const transcription = sttResponse.transcript

  if (!transcription?.trim()) {
    return c.json({ error: 'Could not transcribe audio' }, 422)
  }

  // Gemini: generate response
  const history: { role: string; parts: { text: string }[] }[] =
    JSON.parse(historyJson)

  const geminiHistory = history.map((msg) => ({
    role: msg.role as 'user' | 'model',
    parts: msg.parts,
  }))

  const chat = genai.chats.create({
    model: 'gemini-3.1-flash-lite-preview',
    config: { systemInstruction: systemPrompt },
    history: geminiHistory,
  })

  const result = await chat.sendMessage({ message: transcription })
  const responseText = result.text ?? ''

  // TTS: convert response to speech
  const ttsResponse = await sarvam.textToSpeech.convert({
    text: responseText.slice(0, 2500), // max 2500 chars for bulbul:v3
    target_language_code: languageCode as any,
    model: 'bulbul:v3' as any,
    speaker: 'priya' as any,
  })

  return c.json({
    transcription,
    response: responseText,
    audio: ttsResponse.audios[0],
    detected_language: sttResponse.language_code,
  })
})

export default {
  port: process.env.PORT || 4000,
  fetch: app.fetch,
}
