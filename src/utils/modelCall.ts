/**
 * Shared model call helper: single retry+backoff loop used by the nightly
 * memory jobs (weekly/monthly rollup, diary writer, candidate judge).
 *
 * Returns the raw model text (content, falling back to reasoning_content).
 * Callers keep their own JSON parsing, result normalization and model-name
 * resolution. Throws after the final retry.
 */
import { callOpenAICompat } from "../proxy/openaiAdapter";
import type { Env, OpenAIChatRequest, OpenAIChatResponse } from "../types";

export const MODEL_RETRY_BACKOFF_MS = [2000, 8000];

const DEFAULT_SYSTEM_PROMPT = "你是严格的 JSON 生成器。你只输出 JSON，不要输出思考过程。";

/**
 * DeepSeek V4 models (deepseek-flash / deepseek-v4-pro, reached via the `deepseek/` author
 * prefix through AI Gateway BYOK) run with thinking mode ENABLED BY DEFAULT at `high` effort.
 * For our nightly JSON-generator calls (diary writer, dream extract, rollups) that means the
 * whole `max_tokens` budget is spent on `reasoning_content` and `content` comes back empty with
 * `finish_reason:length` — surfacing as `model_invalid_json` (observed 2026-10-22, Retry 3).
 *
 * The OpenAI-format control to turn thinking off is `reasoning_effort:"none"` (per DeepSeek's
 * official thinking-mode guide: effort `none` disables thinking mode). It is a plain top-level
 * field, so it survives the compat adapter (only Anthropic-native `thinking` is stripped there).
 * We add it ONLY for `deepseek/` models; every other model's request is byte-identical to before.
 */
function isDeepSeekModel(model: string): boolean {
  return model.trim().toLowerCase().startsWith("deepseek/");
}

export interface ModelCallRequest {
  model: string;
  prompt: string;
  maxTokens?: number;
  backoffMs?: number[]; // default [2000, 8000]; pass [] for a single attempt
  systemPrompt?: string; // default JSON-generator system prompt
  logPrefix?: string; // default "model_call"
  logMeta?: Record<string, unknown>;
}

export class ModelCallError extends Error {
  readonly status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "ModelCallError";
    this.status = status;
  }
}

/**
 * Read the first non-empty string env var from `keys`, else `fallback`.
 * Callers still own which keys/fallback to use — the original model readers
 * were near-identical but not byte-identical, so only this lookup pattern is unified.
 */
export function readModelName(env: Env, keys: string[], fallback: string): string {
  const record = env as unknown as Record<string, unknown>;
  for (const key of keys) {
    const raw = record[key];
    if (typeof raw === "string" && raw.trim()) return raw.trim();
  }
  return fallback;
}

/**
 * Default fallback chain for the diary writer and dream (nightly extraction) model calls.
 * The primary (@cf/openai/gpt-oss-120b) returns HTTP 500 in bursts regardless of input size,
 * so on a 5xx / timeout / model_error we retry the SAME call with the next model here. These
 * are all currently-available Workers AI chat models with good Chinese writing and enough
 * context for a ~40-message chunk. The primary itself is NOT listed — it is always tried first
 * by the caller; this is only the ordered list of what to fall back TO. Overridable at runtime
 * via the MODEL_FALLBACKS setting (comma-separated, same `workers-ai/@cf/...` form).
 */
export const DEFAULT_MODEL_FALLBACKS = [
  "workers-ai/@cf/qwen/qwen3-30b-a3b-fp8",
  "workers-ai/@cf/zai-org/glm-4.7-flash",
  "workers-ai/@cf/meta/llama-3.3-70b-instruct-fp8-fast"
];

/**
 * Ordered fallback model list. Reads the MODEL_FALLBACKS setting (comma-separated); blanks and
 * duplicates are dropped. Falls back to DEFAULT_MODEL_FALLBACKS when unset/empty so the chain
 * works out of the box. Returns a fresh array the caller may filter against its primary.
 */
export function readModelFallbacks(env: Env): string[] {
  const raw = env.MODEL_FALLBACKS;
  if (typeof raw === "string" && raw.trim()) {
    const parsed = raw
      .split(",")
      .map((entry) => entry.trim())
      .filter((entry) => entry.length > 0);
    const unique: string[] = [];
    const seen = new Set<string>();
    for (const entry of parsed) {
      if (!seen.has(entry)) {
        seen.add(entry);
        unique.push(entry);
      }
    }
    if (unique.length > 0) return unique;
  }
  return [...DEFAULT_MODEL_FALLBACKS];
}

/**
 * Build the ordered model chain to try: the primary first, then each configured fallback that
 * is not already the primary. Deduped, order-preserving. A single-element result (primary only)
 * means no distinct fallback is configured and behaviour is identical to the pre-fallback path.
 */
export function buildModelChain(primary: string, env: Env): string[] {
  const chain = [primary];
  const seen = new Set<string>([primary]);
  for (const candidate of readModelFallbacks(env)) {
    if (!seen.has(candidate)) {
      seen.add(candidate);
      chain.push(candidate);
    }
  }
  return chain;
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetriableModelStatus(status: number): boolean {
  return status === 429 || status >= 500;
}

export async function callModelWithRetry(env: Env, req: ModelCallRequest): Promise<string> {
  const backoffMs = req.backoffMs ?? MODEL_RETRY_BACKOFF_MS;
  const logPrefix = req.logPrefix ?? "model_call";
  const maxAttempts = 1 + backoffMs.length;
  const startedAt = Date.now();

  const request: OpenAIChatRequest = {
    model: req.model,
    messages: [
      { role: "system", content: req.systemPrompt ?? DEFAULT_SYSTEM_PROMPT },
      { role: "user", content: req.prompt }
    ],
    temperature: 0,
    ...(req.maxTokens !== undefined ? { max_tokens: req.maxTokens } : {}),
    ...(isDeepSeekModel(req.model) ? { reasoning_effort: "none" } : {}),
    response_format: { type: "json_object" },
    stream: false
  };

  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    if (attempt > 0) {
      const waitMs = backoffMs[attempt - 1] ?? backoffMs.at(-1) ?? 8000;
      console.warn(`${logPrefix}: retrying model call after non-ok response`, {
        ...req.logMeta,
        model: req.model,
        attempt: attempt + 1,
        maxAttempts,
        backoffMs: waitMs
      });
      await delay(waitMs);
    }

    try {
      const response = await callOpenAICompat(env, request);
      const elapsedMs = Date.now() - startedAt;
      if (!response.ok) {
        const retriable = isRetriableModelStatus(response.status);
        console.error(`${logPrefix}: model returned non-ok`, {
          ...req.logMeta,
          model: req.model,
          status: response.status,
          statusText: response.statusText,
          elapsedMs,
          attempt: attempt + 1,
          retriable
        });
        if (retriable && attempt < maxAttempts - 1) continue;
        throw new ModelCallError(`model returned status ${response.status}`, response.status);
      }

      const parsed = (await response.json()) as OpenAIChatResponse;
      const message = parsed.choices?.[0]?.message as
        | ({ content?: unknown; reasoning_content?: unknown })
        | undefined;
      const content = typeof message?.content === "string" ? message.content.trim() : "";
      const reasoning = typeof message?.reasoning_content === "string" ? message.reasoning_content.trim() : "";
      return content || reasoning;
    } catch (error) {
      if (error instanceof ModelCallError) throw error;
      const elapsedMs = Date.now() - startedAt;
      const message = error instanceof Error && error.message ? error.message : String(error);
      console.error(`${logPrefix} model failed`, {
        ...req.logMeta,
        model: req.model,
        elapsedMs,
        attempt: attempt + 1,
        error: message
      });
      if (attempt < maxAttempts - 1) continue;
      throw error;
    }
  }

  throw new ModelCallError("model call failed after retries");
}

export interface ModelChainCallRequest extends Omit<ModelCallRequest, "model"> {
  /** Ordered models to try; each gets ONE full callModelWithRetry cycle before the next. */
  models: string[];
}

export interface ModelChainResult {
  /** Raw model text (content, falling back to reasoning_content). */
  text: string;
  /** The model that actually produced the text — store it so stats show which one ran. */
  model: string;
}

/**
 * Try an ordered chain of models for ONE logical call. Each model gets a full
 * `callModelWithRetry` cycle (its existing per-model backoff retry); on a `ModelCallError`
 * — the transport / 5xx / 429 class that the primary (@cf/openai/gpt-oss-120b) throws in
 * bursts — we advance to the next model in the chain. Bounded: each model is tried at most
 * once (after its own existing retry). The last model's error propagates if the whole chain
 * is exhausted, so callers keep their current `model_error` handling unchanged. Returns the
 * text AND the model that produced it so the diary/dream stats can record the real author.
 *
 * A single-element `models` behaves exactly like `callModelWithRetry` (plus the model in the
 * return value), so wiring this in never changes behaviour when no distinct fallback exists.
 */
export async function callModelChainWithRetry(env: Env, req: ModelChainCallRequest): Promise<ModelChainResult> {
  const { models, ...rest } = req;
  if (models.length === 0) throw new ModelCallError("model chain is empty");
  const logPrefix = rest.logPrefix ?? "model_call";
  let lastError: unknown;

  for (let i = 0; i < models.length; i += 1) {
    const model = models[i];
    const isLast = i === models.length - 1;
    try {
      const text = await callModelWithRetry(env, { ...rest, model });
      if (i > 0) {
        console.warn(`${logPrefix}: fallback model produced output`, {
          ...rest.logMeta,
          model,
          chainIndex: i,
          chainLength: models.length
        });
      }
      return { text, model };
    } catch (error) {
      lastError = error;
      // ModelCallError is the retriable transport/5xx/429 class — advance to the next model.
      // Any other thrown error is also treated as a failed attempt for this model.
      console.error(`${logPrefix}: model failed, ${isLast ? "chain exhausted" : "advancing to next fallback"}`, {
        ...rest.logMeta,
        model,
        chainIndex: i,
        chainLength: models.length,
        status: error instanceof ModelCallError ? error.status : undefined
      });
      if (isLast) throw error;
    }
  }

  // Unreachable (the last iteration always returns or throws), but keeps the type checker happy.
  throw (lastError instanceof Error ? lastError : new ModelCallError("model chain failed"));
}
