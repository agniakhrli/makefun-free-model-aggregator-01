import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicEventsToOpenAI,
  anthropicModelList,
  anthropicResponseToOpenAI,
  anthropicToOpenAI,
  geminiEventsToAnthropic,
  geminiEventsToOpenAI,
  geminiResponseToAnthropic,
  geminiResponseToOpenAI,
  openAIError,
  openAIModelList,
  openAIRequestToAnthropic,
  openAIToAnthropic,
  requestToGemini,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;

const providerCalls = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

function candidatesFor(requires: Requirement, exhausted: Set<string>): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (exhausted.has(probe.provider.id)) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const rating = row?.rating_milli ?? 0;
      out.push({ probe, model, rating });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    const ctxA = a.model.context_length ?? 0;
    const ctxB = b.model.context_length ?? 0;
    if (ctxA !== ctxB) return ctxB - ctxA;
    const countA = providerCalls.get(a.probe.provider.id) ?? 0;
    const countB = providerCalls.get(b.probe.provider.id) ?? 0;
    if (countA !== countB) return countA - countB;
    return a.probe.provider.id.localeCompare(b.probe.provider.id);
  });
  return out;
}

function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

async function invokeUpstream(
  clientApi: 'openai' | 'anthropic',
  body: Json,
  isStreaming: boolean,
  requires: Requirement
): Promise<{ upstream: Response; candidate: Candidate; callsCost: number } | null> {
  const exhausted = new Set<string>();
  let callsCost = 0;

  for (;;) {
    const candidates = candidatesFor(requires, exhausted);
    if (candidates.length === 0) break;
    const candidate = candidates[0];

    callsCost += 1;
    providerCalls.set(candidate.probe.provider.id, (providerCalls.get(candidate.probe.provider.id) ?? 0) + 1);

    const base = joinBase(ENV_URL, candidate.probe.provider.base);
    const wire = candidate.probe.provider.wire;
    const key = candidate.probe.provider.key;
    const auth = candidate.probe.provider.auth;

    let url = '';
    let reqHeaders: Record<string, string> = {
      'content-type': 'application/json',
      ...authHeaders(auth, key),
    };
    let upstreamBody: Json = {};

    if (wire === 'openai') {
      url = `${base}/v1/chat/completions`;
      if (clientApi === 'openai') {
        upstreamBody = { ...body, model: candidate.model.id };
      } else {
        upstreamBody = anthropicToOpenAI(body, candidate.model.id);
      }
      if (isStreaming) {
        reqHeaders.accept = 'text/event-stream';
        upstreamBody.stream = true;
        upstreamBody.stream_options = { include_usage: true };
      }
    } else if (wire === 'anthropic') {
      url = `${base}/v1/messages`;
      reqHeaders['anthropic-version'] = '2023-06-01';
      if (clientApi === 'anthropic') {
        upstreamBody = { ...body, model: candidate.model.id };
      } else {
        upstreamBody = openAIRequestToAnthropic(body, candidate.model.id);
      }
      if (isStreaming) {
        reqHeaders.accept = 'text/event-stream';
        upstreamBody.stream = true;
      }
    } else if (wire === 'gemini') {
      const action = isStreaming ? 'streamGenerateContent?alt=sse' : 'generateContent';
      url = `${base}/v1beta/models/${candidate.model.id}:${action}`;
      if (auth === 'x-goog-api-key' && key !== '') {
        reqHeaders['x-goog-api-key'] = key;
      }
      upstreamBody = requestToGemini(body, candidate.model.id);
      if (isStreaming) reqHeaders.accept = 'text/event-stream';
    }

    let upstream: Response;
    try {
      upstream = await fetch(url, {
        method: 'POST',
        headers: reqHeaders,
        body: JSON.stringify(upstreamBody),
      });
    } catch {
      exhausted.add(candidate.probe.provider.id);
      continue;
    }

    if (upstream.status === 402 || upstream.status === 429 || upstream.status >= 500) {
      exhausted.add(candidate.probe.provider.id);
      try { await upstream.arrayBuffer(); } catch {}
      continue;
    }

    return { upstream, candidate, callsCost };
  }

  return null;
}

async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const isStreaming = body.stream === true;
  const result = await invokeUpstream('openai', body, isStreaming, requirementsFor(body));
  if (result === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');

  const { upstream, candidate } = result;
  const wire = candidate.probe.provider.wire;
  const requestedModel = typeof body.model === 'string' && body.model !== '' ? body.model : candidate.model.id;

  if (isStreaming && upstream.ok && upstream.body !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    if (wire === 'openai') {
      await pipeStream(upstream.body, res);
    } else if (wire === 'anthropic') {
      for await (const event of anthropicEventsToOpenAI(upstream.body, requestedModel)) res.write(event);
      res.end();
    } else if (wire === 'gemini') {
      for await (const event of geminiEventsToOpenAI(upstream.body, requestedModel)) res.write(event);
      res.end();
    }
    return;
  }

  const text = await upstream.text();
  let jsonAnswer: unknown;
  try {
    jsonAnswer = JSON.parse(text);
  } catch {
    res.writeHead(upstream.status, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
    res.end(text);
    return;
  }

  if (wire === 'openai') {
    sendJson(res, upstream.status, jsonAnswer);
  } else if (wire === 'anthropic') {
    sendJson(res, upstream.status, anthropicResponseToOpenAI(isRecord(jsonAnswer) ? jsonAnswer : {}, requestedModel));
  } else if (wire === 'gemini') {
    sendJson(res, upstream.status, geminiResponseToOpenAI(isRecord(jsonAnswer) ? jsonAnswer : {}, requestedModel));
  }
}

async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const isStreaming = body.stream === true;
  const result = await invokeUpstream('anthropic', body, isStreaming, requirementsFor(body));
  if (result === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');

  const { upstream, candidate } = result;
  const wire = candidate.probe.provider.wire;
  const requestedModel = typeof body.model === 'string' && body.model !== '' ? body.model : candidate.model.id;

  if (!upstream.ok) {
    const text = await upstream.text();
    return sendJson(res, upstream.status, anthropicError('api_error', `the provider answered HTTP ${upstream.status}: ${text.slice(0, 500)}`));
  }

  if (isStreaming && upstream.body !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    if (wire === 'anthropic') {
      await pipeStream(upstream.body, res);
    } else if (wire === 'openai') {
      for await (const event of anthropicEventsFromOpenAI(upstream.body, requestedModel)) res.write(event);
      res.end();
    } else if (wire === 'gemini') {
      for await (const event of geminiEventsToAnthropic(upstream.body, requestedModel)) res.write(event);
      res.end();
    }
    return;
  }

  const answer: unknown = await upstream.json();
  const recAnswer = isRecord(answer) ? answer : {};
  if (wire === 'anthropic') {
    sendJson(res, 200, recAnswer);
  } else if (wire === 'openai') {
    sendJson(res, 200, openAIToAnthropic(recAnswer, requestedModel));
  } else if (wire === 'gemini') {
    sendJson(res, 200, geminiResponseToAnthropic(recAnswer, requestedModel));
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // Try again on next tick
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
