// The client wires and the one upstream wire this baseline speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This baseline forwards both to a
// provider on the openai wire and translates back through a unified translation layer.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter' || finish === 'refusal') return 'refusal';
  if (finish === 'stop') return 'end_turn';
  return typeof finish === 'string' && finish !== '' ? finish : 'end_turn';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

// --- Canonical Intermediate Representations ---

export interface CanonicalToolCall {
  id: string;
  name: string;
  input: Json;
}

export interface CanonicalTool {
  name: string;
  description: string;
  input_schema: Json;
}

export interface CanonicalMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | Json[];
  tool_calls?: CanonicalToolCall[];
  tool_call_id?: string;
}

export interface CanonicalRequest {
  model: string;
  system: string;
  messages: CanonicalMessage[];
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop_sequences?: string[];
  tools?: CanonicalTool[];
  tool_choice?: unknown;
  stream?: boolean;
}

export interface CanonicalResponse {
  id: string;
  model: string;
  text: string;
  tool_calls: CanonicalToolCall[];
  stop_reason: string;
  usage: {
    input_tokens: number;
    output_tokens: number;
  };
}

// --- Parsers (Wire -> Canonical) ---

export function parseAnthropicRequest(body: Json, model: string): CanonicalRequest {
  const messages: CanonicalMessage[] = [];
  const system = textOf(body.system);
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const tool_calls: CanonicalToolCall[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        tool_calls.push({
          id: typeof block.id === 'string' ? block.id : '',
          name: typeof block.name === 'string' ? block.name : '',
          input: isRecord(block.input) ? block.input : {},
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : '',
        ...(tool_calls.length > 0 ? { tool_calls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: CanonicalRequest = { model, system, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences)) out.stop_sequences = body.stop_sequences.filter((s): s is string => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    out.tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        input_schema: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      }];
    });
  }
  if (body.tool_choice !== undefined) out.tool_choice = body.tool_choice;
  if (body.stream === true) out.stream = true;
  return out;
}

export function parseOpenAIRequest(body: Json, model: string): CanonicalRequest {
  const messages: CanonicalMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  let system = '';
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const t = textOf(raw.content);
      if (t !== '') system = system ? `${system}\n${t}` : t;
      continue;
    }
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'tool',
        tool_call_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
        content: toolResultText(raw.content),
      });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const tool_calls: CanonicalToolCall[] = [];
      const toolCallsRaw = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of toolCallsRaw) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        tool_calls.push({
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({
        role: 'assistant',
        content: textOf(content),
        ...(tool_calls.length > 0 ? { tool_calls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') {
        parts.push({ type: 'text', text: part.text });
      } else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
        parts.push({ type: 'image_url', image_url: part.image_url });
      }
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: CanonicalRequest = { model, system, messages };
  const maxTokens = numberOrZero(body.max_tokens) || numberOrZero(body.max_completion_tokens);
  if (maxTokens > 0) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop)) {
    out.stop_sequences = body.stop.filter((s): s is string => typeof s === 'string');
  } else if (typeof body.stop === 'string' && body.stop !== '') {
    out.stop_sequences = [body.stop];
  }
  if (Array.isArray(body.tools)) {
    out.tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      }];
    });
  }
  if (body.tool_choice !== undefined) out.tool_choice = body.tool_choice;
  if (body.stream === true) out.stream = true;
  return out;
}

export function parseAnthropicResponse(body: Json, model: string): CanonicalResponse {
  const content = body.content;
  let text = '';
  const tool_calls: CanonicalToolCall[] = [];
  if (Array.isArray(content)) {
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') {
        text += block.text;
      } else if (block.type === 'tool_use') {
        tool_calls.push({
          id: typeof block.id === 'string' ? block.id : 'toolu_0',
          name: typeof block.name === 'string' ? block.name : '',
          input: isRecord(block.input) ? block.input : {},
        });
      }
    }
  } else if (typeof content === 'string') {
    text = content;
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const stop_reason = stopReasonOf(body.stop_reason);
  return {
    id: typeof body.id === 'string' ? body.id : 'aggregate',
    model,
    text,
    tool_calls,
    stop_reason,
    usage: {
      input_tokens: numberOrZero(usage.input_tokens),
      output_tokens: numberOrZero(usage.output_tokens),
    },
  };
}

export function parseOpenAIResponse(body: Json, model: string): CanonicalResponse {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  let text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const tool_calls: CanonicalToolCall[] = [];
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    tool_calls.push({
      id: typeof call.id === 'string' ? call.id : 'call_0',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const stop_reason = stopReasonOf(choice.finish_reason);
  return {
    id: typeof body.id === 'string' ? body.id : 'aggregate',
    model,
    text,
    tool_calls,
    stop_reason,
    usage: {
      input_tokens: numberOrZero(usage.prompt_tokens),
      output_tokens: numberOrZero(usage.completion_tokens),
    },
  };
}

export function parseGeminiResponse(body: Json, model: string): CanonicalResponse {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  for (const part of parts) {
    if (isRecord(part) && typeof part.text === 'string') text += part.text;
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    id: 'gemini',
    model,
    text,
    tool_calls: [],
    stop_reason: 'end_turn',
    usage: {
      input_tokens: numberOrZero(usage.promptTokenCount),
      output_tokens: numberOrZero(usage.candidatesTokenCount),
    },
  };
}

// --- Renderers (Canonical -> Wire) ---

export function renderOpenAIRequest(req: CanonicalRequest): Json {
  const messages: Json[] = [];
  if (req.system !== '') {
    messages.push({ role: 'system', content: req.system });
  }
  for (const m of req.messages) {
    if (m.role === 'tool') {
      messages.push({
        role: 'tool',
        tool_call_id: m.tool_call_id ?? '',
        content: toolResultText(m.content),
      });
      continue;
    }
    if (m.role === 'assistant') {
      const tool_calls = m.tool_calls?.map((tc) => ({
        id: tc.id,
        type: 'function',
        function: {
          name: tc.name,
          arguments: JSON.stringify(tc.input),
        },
      }));
      messages.push({
        role: 'assistant',
        content: typeof m.content === 'string' && m.content !== '' ? m.content : null,
        ...(tool_calls && tool_calls.length > 0 ? { tool_calls } : {}),
      });
      continue;
    }
    messages.push({ role: m.role, content: m.content });
  }

  const out: Json = { model: req.model, messages };
  if (req.max_tokens !== undefined) out.max_tokens = req.max_tokens;
  if (req.temperature !== undefined) out.temperature = req.temperature;
  if (req.top_p !== undefined) out.top_p = req.top_p;
  if (req.stop_sequences && req.stop_sequences.length > 0) out.stop = req.stop_sequences;
  if (req.tools && req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({
      type: 'function',
      function: {
        name: t.name,
        description: t.description,
        parameters: t.input_schema,
      },
    }));
  }
  if (req.tool_choice !== undefined) {
    const choice = req.tool_choice;
    if (isRecord(choice)) {
      if (choice.type === 'auto') out.tool_choice = 'auto';
      else if (choice.type === 'any') out.tool_choice = 'required';
      else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
      else out.tool_choice = choice;
    } else {
      out.tool_choice = choice;
    }
  }
  if (req.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

export function renderAnthropicRequest(req: CanonicalRequest): Json {
  const messages: Json[] = [];
  for (const m of req.messages) {
    if (m.role === 'system') continue;
    if (m.role === 'assistant') {
      const content: Json[] = [];
      if (typeof m.content === 'string' && m.content !== '') {
        content.push({ type: 'text', text: m.content });
      }
      if (m.tool_calls) {
        for (const tc of m.tool_calls) {
          content.push({
            type: 'tool_use',
            id: tc.id,
            name: tc.name,
            input: tc.input,
          });
        }
      }
      if (content.length === 0) content.push({ type: 'text', text: '' });
      messages.push({ role: 'assistant', content });
      continue;
    }
    if (m.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{
          type: 'tool_result',
          tool_use_id: m.tool_call_id ?? '',
          content: toolResultText(m.content),
        }],
      });
      continue;
    }
    messages.push({ role: 'user', content: m.content });
  }

  const out: Json = { model: req.model, messages };
  if (req.system !== '') out.system = req.system;
  const maxTokens = req.max_tokens ?? 4096;
  out.max_tokens = maxTokens > 0 ? maxTokens : 4096;
  if (req.temperature !== undefined) out.temperature = req.temperature;
  if (req.top_p !== undefined) out.top_p = req.top_p;
  if (req.stop_sequences && req.stop_sequences.length > 0) out.stop_sequences = req.stop_sequences;
  if (req.tools && req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({
      name: t.name,
      description: t.description,
      input_schema: t.input_schema,
    }));
  }
  if (req.stream === true) out.stream = true;
  return out;
}

export function renderGeminiRequest(req: CanonicalRequest): Json {
  let systemText = req.system;
  const contents: Json[] = [];
  for (const m of req.messages) {
    if (m.role === 'system') {
      const t = textOf(m.content);
      if (t !== '') systemText = systemText ? `${systemText}\n${t}` : t;
      continue;
    }
    const role = m.role === 'assistant' ? 'model' : 'user';
    const content = m.content;
    const parts: Json[] = [];
    if (typeof content === 'string') {
      parts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') {
          parts.push({ text: block.text });
        } else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          const url = block.image_url.url;
          const match = /^data:([^;]+);base64,(.+)$/.exec(url);
          if (match) {
            parts.push({ inlineData: { mimeType: match[1], data: match[2] } });
          } else {
            parts.push({ fileData: { fileUri: url, mimeType: 'image/jpeg' } });
          }
        } else if (block.type === 'image' && isRecord(block.source)) {
          const source = block.source;
          if (source.type === 'base64' && typeof source.data === 'string') {
            parts.push({ inlineData: { mimeType: source.media_type ?? 'image/png', data: source.data } });
          } else if (source.type === 'url' && typeof source.url === 'string') {
            parts.push({ fileData: { fileUri: source.url, mimeType: 'image/jpeg' } });
          }
        }
      }
    }
    if (m.tool_calls) {
      for (const tc of m.tool_calls) {
        parts.push({ functionCall: { name: tc.name, args: tc.input } });
      }
    }
    if (parts.length > 0) {
      contents.push({ role, parts });
    }
  }

  const out: Json = { contents };
  if (systemText !== '') {
    out.systemInstruction = { parts: [{ text: systemText }] };
  }
  const maxTokens = req.max_tokens ?? 0;
  const stopSequences = req.stop_sequences ?? [];
  if (maxTokens > 0 || stopSequences.length > 0) {
    out.generationConfig = {
      ...(maxTokens > 0 ? { maxOutputTokens: maxTokens } : {}),
      ...(stopSequences.length > 0 ? { stopSequences } : {}),
    };
  }
  return out;
}

export function renderOpenAIResponse(resp: CanonicalResponse): Json {
  let finishReason = 'stop';
  if (resp.stop_reason === 'max_tokens') finishReason = 'length';
  else if (resp.stop_reason === 'tool_use') finishReason = 'tool_calls';
  else if (resp.stop_reason === 'refusal') finishReason = 'refusal';

  const tool_calls = resp.tool_calls.map((tc) => ({
    id: tc.id,
    type: 'function',
    function: {
      name: tc.name,
      arguments: JSON.stringify(tc.input),
    },
  }));

  return {
    id: `chatcmpl-${resp.id}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model: resp.model,
    choices: [{
      index: 0,
      message: {
        role: 'assistant',
        content: resp.text !== '' ? resp.text : null,
        ...(tool_calls.length > 0 ? { tool_calls } : {}),
      },
      finish_reason: finishReason,
    }],
    usage: {
      prompt_tokens: resp.usage.input_tokens,
      completion_tokens: resp.usage.output_tokens,
      total_tokens: resp.usage.input_tokens + resp.usage.output_tokens,
    },
  };
}

export function renderAnthropicResponse(resp: CanonicalResponse): Json {
  const content: Json[] = [];
  if (resp.text !== '') {
    content.push({ type: 'text', text: resp.text });
  }
  for (const tc of resp.tool_calls) {
    content.push({
      type: 'tool_use',
      id: tc.id,
      name: tc.name,
      input: tc.input,
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });

  let stopReason = 'end_turn';
  if (resp.stop_reason === 'length' || resp.stop_reason === 'max_tokens') stopReason = 'max_tokens';
  else if (resp.stop_reason === 'tool_calls' || resp.stop_reason === 'tool_use') stopReason = 'tool_use';

  return {
    id: `msg_${resp.id}`,
    type: 'message',
    role: 'assistant',
    model: resp.model,
    content,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: resp.usage.input_tokens,
      output_tokens: resp.usage.output_tokens,
    },
  };
}

// --- Public Entrypoint Wrappers ---

export function anthropicToOpenAI(body: Json, model: string): Json {
  const req = parseAnthropicRequest(body, model);
  return renderOpenAIRequest(req);
}

export function openAIRequestToAnthropic(body: Json, model: string): Json {
  const req = parseOpenAIRequest(body, model);
  return renderAnthropicRequest(req);
}

export function openAIToAnthropic(body: Json, model: string): Json {
  const resp = parseOpenAIResponse(body, model);
  return renderAnthropicResponse(resp);
}

export function anthropicResponseToOpenAI(body: Json, model: string): Json {
  const resp = parseAnthropicResponse(body, model);
  return renderOpenAIResponse(resp);
}

export function requestToGemini(body: Json, model: string): Json {
  const req = parseOpenAIRequest(body, model);
  return renderGeminiRequest(req);
}

export function geminiResponseToOpenAI(body: Json, model: string): Json {
  const resp = parseGeminiResponse(body, model);
  return renderOpenAIResponse(resp);
}

export function geminiResponseToAnthropic(body: Json, model: string): Json {
  const resp = parseGeminiResponse(body, model);
  return renderAnthropicResponse(resp);
}

export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

// --- SSE Stream Translators ---

export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };

  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    }
    toolIndexes.clear();
    return out;
  };

  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
        pendingStop = stopReasonOf(choice.finish_reason);
      }
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      for (const event of eventsOfFrame(buffer)) yield event;
    }
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

export async function* anthropicEventsToOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const line of frame.split('\n')) {
          if (line.startsWith('data:')) {
            const dataStr = line.slice(5).trim();
            try {
              const obj = JSON.parse(dataStr);
              if (isRecord(obj)) {
                if (obj.type === 'content_block_delta' && isRecord(obj.delta) && obj.delta.type === 'text_delta') {
                  const chunk = {
                    id: 'chatcmpl-stream',
                    object: 'chat.completion.chunk',
                    created: FIXED_OPENAI_CREATED,
                    model,
                    choices: [{ index: 0, delta: { content: obj.delta.text }, finish_reason: null }],
                  };
                  yield `data: ${JSON.stringify(chunk)}\n\n`;
                } else if (obj.type === 'message_delta' && isRecord(obj.delta)) {
                  const stopReason = obj.delta.stop_reason === 'max_tokens' ? 'length' : 'stop';
                  const chunk = {
                    id: 'chatcmpl-stream',
                    object: 'chat.completion.chunk',
                    created: FIXED_OPENAI_CREATED,
                    model,
                    choices: [{ index: 0, delta: {}, finish_reason: stopReason }],
                    ...(isRecord(obj.usage) ? { usage: { prompt_tokens: 0, completion_tokens: numberOrZero(obj.usage.output_tokens), total_tokens: numberOrZero(obj.usage.output_tokens) } } : {}),
                  };
                  yield `data: ${JSON.stringify(chunk)}\n\n`;
                }
              }
            } catch {}
          }
        }
        at = buffer.indexOf('\n\n');
      }
    }
  } finally {
    reader.releaseLock();
  }
  yield 'data: [DONE]\n\n';
}

export async function* geminiEventsToOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const line of frame.split('\n')) {
          if (line.startsWith('data:')) {
            const dataStr = line.slice(5).trim();
            try {
              const obj = JSON.parse(dataStr);
              if (isRecord(obj) && Array.isArray(obj.candidates)) {
                const candidate = isRecord(obj.candidates[0]) ? obj.candidates[0] : {};
                const content = isRecord(candidate.content) ? candidate.content : {};
                const parts = Array.isArray(content.parts) ? content.parts : [];
                for (const part of parts) {
                  if (isRecord(part) && typeof part.text === 'string' && part.text !== '') {
                    const chunk = {
                      id: 'chatcmpl-gemini',
                      object: 'chat.completion.chunk',
                      created: FIXED_OPENAI_CREATED,
                      model,
                      choices: [{ index: 0, delta: { content: part.text }, finish_reason: null }],
                    };
                    yield `data: ${JSON.stringify(chunk)}\n\n`;
                  }
                }
                if (candidate.finishReason) {
                  const chunk = {
                    id: 'chatcmpl-gemini',
                    object: 'chat.completion.chunk',
                    created: FIXED_OPENAI_CREATED,
                    model,
                    choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
                  };
                  yield `data: ${JSON.stringify(chunk)}\n\n`;
                }
              }
            } catch {}
          }
        }
        at = buffer.indexOf('\n\n');
      }
    }
  } finally {
    reader.releaseLock();
  }
  yield 'data: [DONE]\n\n';
}

export async function* geminiEventsToAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  yield sseEvent('message_start', {
    type: 'message_start',
    message: { id: 'msg_gemini', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
  });
  yield sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const line of frame.split('\n')) {
          if (line.startsWith('data:')) {
            const dataStr = line.slice(5).trim();
            try {
              const obj = JSON.parse(dataStr);
              if (isRecord(obj) && Array.isArray(obj.candidates)) {
                const candidate = isRecord(obj.candidates[0]) ? obj.candidates[0] : {};
                const content = isRecord(candidate.content) ? candidate.content : {};
                const parts = Array.isArray(content.parts) ? content.parts : [];
                for (const part of parts) {
                  if (isRecord(part) && typeof part.text === 'string' && part.text !== '') {
                    yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: part.text } });
                  }
                }
              }
            } catch {}
          }
        }
        at = buffer.indexOf('\n\n');
      }
    }
  } finally {
    reader.releaseLock();
  }
  yield sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: 0 } });
  yield sseEvent('message_stop', { type: 'message_stop' });
}
