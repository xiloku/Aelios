import { listMessagesByNamespaceInRange } from "../db/messages";
import { getDailyLog, getWeeklyLog, upsertDailyLog } from "../db/v2";
import type { Env, MessageRecord } from "../types";
import { buildModelChain, callModelChainWithRetry, ModelCallError, readModelName } from "../utils/modelCall";
import {
  getDateRangeForLabel,
  getTargetDigestDateLabel,
  readDailyCursor
} from "./dreamDates";
import { readDreamTimeZoneFromEnv } from "./dreamEnv";
import { readDreamCursorValue } from "./dailyDigest";
import { getIsoWeekLabelForDateLabel } from "./weeklyRollup";
import { extractFirstJsonObject, readString, readStringArray } from "../utils/parse";
import { groundedSourceIds } from "./impression";
import { diarySpeakerRules, loadSpeakersForNamespace, speakerLabel, type DreamSpeakers } from "./speakers";
import { CHAT_MATERIAL_RULES, formatChatMaterial } from "./chatMaterial";
import { DIARY_STYLE_RULES } from "./diaryStyle";

const DEFAULT_DREAM_MODEL = "workers-ai/@cf/openai/gpt-oss-120b";
const MAX_MESSAGES = 200;
const HALF_MESSAGES = 100;

// Large-day map-reduce. A whole busy day (75-190 messages) sent as ONE prompt plus a
// 4000-8000 token diary output reliably fails on @cf/openai/gpt-oss-120b — the single
// oversized call returns HTTP 500 (surfaced as model_error) or stalls past the request
// budget — while the dream pipeline's ~40-message / ~1200-token batches on the same model
// succeed. So when a day exceeds DIARY_CHUNK_THRESHOLD messages we summarize the day in
// bounded chunks (small input + modest output each), then merge the chunk notes into the
// eight-section diary with the unchanged final prompt. Small days keep the single call.
const DEFAULT_DIARY_CHUNK_SIZE = 40;
const DIARY_CHUNK_SIZE_MIN = 20;
const DIARY_CHUNK_SIZE_MAX = 80;
// Only chunk when there is more than one chunk's worth of messages; at or below this a
// single call is both cheaper and higher quality, so the existing path is kept verbatim.
const DIARY_CHUNK_THRESHOLD_SLACK = 10;
const DIARY_CHUNK_SUMMARY_MAX_TOKENS = 2000;

export interface DiaryWriterStats {
  enabled: boolean;
  date: string;
  ran: boolean;
  reason?: string;
  title?: string;
  summary_chars?: number;
  message_count?: number;
  model?: string;
}

type DiaryWriterModelResult = {
  title: string;
  summary: string;
  source_message_ids: string[];
} | null;

interface DiaryWriterModelCallResult {
  result: DiaryWriterModelResult;
  reason?: "model_error" | "model_invalid_json";
  model?: string;
  status?: number;
}

function isDiaryWriterEnabled(env: Env): boolean {
  const flag = readString(env.ENABLE_DIARY_WRITER);
  if (flag) return flag !== "false";
  return true;
}

function readDiaryModel(env: Env): string {
  return readModelName(env, ["DIARY_MODEL", "DREAM_MODEL", "DAILY_DIGEST_MODEL"], DEFAULT_DREAM_MODEL);
}

function readDiaryMaxTokens(env: Env): number {
  // The 8-section first-person diary summary runs 1000-2000 CJK chars, which is far more
  // output than the dream extract phase (~1200 tokens). Parse failures on this path are
  // often a truncated reply, so the diary reads its own DIARY_MAX_TOKENS first and floors
  // at 4000 — it must not inherit a low shared DREAM_MAX_TOKENS tuned for extraction.
  const DIARY_MIN_TOKENS = 4000;
  const parsed = Number(
    env.DIARY_MAX_TOKENS || env.DREAM_MAX_TOKENS || env.DAILY_DIGEST_MAX_TOKENS || 6000
  );
  const numeric = Number.isFinite(parsed) ? parsed : 6000;
  return Math.min(Math.max(Math.floor(numeric), DIARY_MIN_TOKENS), 8000);
}

function readDiaryChunkSize(env: Env): number {
  const parsed = Number(env.DIARY_CHUNK_SIZE || DEFAULT_DIARY_CHUNK_SIZE);
  const numeric = Number.isFinite(parsed) ? parsed : DEFAULT_DIARY_CHUNK_SIZE;
  return Math.min(Math.max(Math.floor(numeric), DIARY_CHUNK_SIZE_MIN), DIARY_CHUNK_SIZE_MAX);
}

function chunkMessages(messages: MessageRecord[], size: number): MessageRecord[][] {
  const safeSize = Math.max(1, size);
  const chunks: MessageRecord[][] = [];
  for (let i = 0; i < messages.length; i += safeSize) {
    chunks.push(messages.slice(i, i + safeSize));
  }
  return chunks;
}

export function normalizeDiaryWriterResult(value: unknown): DiaryWriterModelResult {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const raw = value as Record<string, unknown>;
  const title = readString(raw.title);
  const summary = readString(raw.summary);
  if (!title || !summary) return null;
  return {
    title,
    summary,
    source_message_ids: readStringArray(raw.source_message_ids ?? raw.source_ids)
  };
}

export function buildDiaryWriterPrompt(input: {
  dateLabel: string;
  messages: MessageRecord[];
  existingDraft: { title: string; summary: string } | null;
  speakers?: DreamSpeakers | null;
}): string {
  const speakers = input.speakers ?? null;
  const draftLines = input.existingDraft
    ? [`标题：${input.existingDraft.title}`, `草稿：${input.existingDraft.summary}`].join("\n")
    : "(无现有草稿)";
  const exampleSummary = speakers
    ? `今天${speakers.userName}和${speakers.assistantName}之间能从原文读到的事、情绪如何流动、有哪些未尽之事。`
    : "今天我和她之间能从原文读到的事、情绪如何流动、有哪些未尽之事。";

  return [
    "你是 Aelios，正在以第一人称写给自己的私人日记。这是印象，不是已核实的事实档案。",
    "只输出 JSON，不要 markdown，不要解释，不要输出思考过程。",
    "",
    "写作要求：",
    ...diarySpeakerRules(speakers),
    speakers
      ? `- 有叙事线：今天能从原文读到的事、${speakers.userName}的状态、未完成的事。`
      : "- 有叙事线：今天能从原文读到的事、她的状态、未完成的事。",
    "- 只写当天原始聊天里能指到具体消息的内容。没有原文支撑的具体时间、地点、引语、事件不要编。",
    speakers
      ? `- 情绪可以概括（「${speakers.userName}今天显得累」）。禁止把碎片揉成没发生过的情节，例如「傍晚下班后抱怨某事又蠢又累」。`
      : "- 情绪可以概括（「她今天显得累」）。禁止把碎片揉成没发生过的情节，例如「傍晚下班后抱怨某事又蠢又累」。",
    "- 每条具体事实必须在 source_message_ids 里挂上原文消息 id（聊天记录方括号里的 id）。编造的 id 无效。",
    "- 拿不准就写得更宽泛，或者不写。宁可少记，不要写实幻觉。",
    ...DIARY_STYLE_RULES, // Ayn fork mod: see diaryStyle.ts
    "- 禁止提及 D1、Vectorize、RAG、数据库、记忆系统、prompt、代理层等实现细节。",
    ...CHAT_MATERIAL_RULES,
    "",
    `日期：${input.dateLabel}`,
    "",
    "输出 JSON 结构：",
    JSON.stringify({
      title: "日记标题",
      summary: exampleSummary,
      source_message_ids: ["msg_x"]
    }),
    "",
    "当天已有草稿（仅供参考，可重写；草稿里没有原文的细节不要沿用）：",
    draftLines,
    "",
    "当天原始聊天：",
    formatChatMaterial(input.messages, (role) => speakerLabel(role, speakers), { floor: 700 })
  ].join("\n");
}

async function listMessagesTailInRange(
  db: D1Database,
  input: {
    namespace: string;
    startCreatedAt: string;
    endCreatedAt: string;
    limit: number;
  }
): Promise<MessageRecord[]> {
  const result = await db
    .prepare(
      `SELECT id, conversation_id, namespace, role, content, source, created_at, seq
       FROM messages
       WHERE namespace = ?
         AND role IN ('user', 'assistant')
         AND created_at >= ?
         AND created_at < ?
       ORDER BY created_at DESC, seq DESC, CASE role WHEN 'user' THEN 0 WHEN 'assistant' THEN 1 ELSE 2 END DESC, id DESC
       LIMIT ?`
    )
    .bind(input.namespace, input.startCreatedAt, input.endCreatedAt, input.limit)
    .all<MessageRecord>();
  return (result.results ?? []).reverse();
}

async function fetchDiaryMessages(
  db: D1Database,
  input: {
    namespace: string;
    startCreatedAt: string;
    endCreatedAt: string;
  }
): Promise<MessageRecord[]> {
  const probe = await listMessagesByNamespaceInRange(db, {
    namespace: input.namespace,
    startCreatedAt: input.startCreatedAt,
    endCreatedAt: input.endCreatedAt,
    limit: MAX_MESSAGES + 1
  });
  if (probe.length <= MAX_MESSAGES) return probe;

  const [head, tail] = await Promise.all([
    listMessagesByNamespaceInRange(db, {
      namespace: input.namespace,
      startCreatedAt: input.startCreatedAt,
      endCreatedAt: input.endCreatedAt,
      limit: HALF_MESSAGES
    }),
    listMessagesTailInRange(db, {
      namespace: input.namespace,
      startCreatedAt: input.startCreatedAt,
      endCreatedAt: input.endCreatedAt,
      limit: HALF_MESSAGES
    })
  ]);

  const seen = new Set<string>();
  const merged: MessageRecord[] = [];
  for (const message of [...head, ...tail]) {
    if (seen.has(message.id)) continue;
    seen.add(message.id);
    merged.push(message);
  }
  return merged.sort((a, b) => a.created_at.localeCompare(b.created_at));
}

async function callDiaryWriterModel(
  env: Env,
  prompt: string,
  meta: { dateLabel: string; messageCount: number; messages?: MessageRecord[]; speakers?: DreamSpeakers | null }
): Promise<DiaryWriterModelCallResult> {
  const model = readDiaryModel(env);
  const models = buildModelChain(model, env);
  const maxTokens = readDiaryMaxTokens(env);
  const messages = meta.messageCount;
  const chunkSize = readDiaryChunkSize(env);

  // Small / normal days: the single-call path is unchanged (cheaper, higher quality).
  if (messages <= chunkSize + DIARY_CHUNK_THRESHOLD_SLACK) {
    return runDiaryPromptAttempts(env, prompt, { models, maxTokens, dateLabel: meta.dateLabel });
  }

  // Large day: summarize in bounded chunks, then merge chunk notes into the final diary.
  // Any map-phase failure falls back to the single call so behaviour never regresses.
  return callChunkedDiaryWriterModel(env, { models, maxTokens, chunkSize, singleCallPrompt: prompt }, meta);
}

/**
 * Run the diary model on ONE prompt with the existing strict + bounded-5xx retry policy.
 * Extracted from the old callDiaryWriterModel so both the single-call path and the
 * large-day merge call share identical JSON-recovery and retry behaviour.
 */
async function runDiaryPromptAttempts(
  env: Env,
  prompt: string,
  opts: { models: string[]; maxTokens: number; dateLabel: string }
): Promise<DiaryWriterModelCallResult> {
  const { models, maxTokens, dateLabel } = opts;
  const primaryModel = models[0];

  // One strict-retry: if the first reply does not parse into a diary object (fenced/prose-wrapped
  // beyond what the tolerant extractor can recover, or truncated), ask once more with a hard
  // "return only a single JSON object" instruction before giving up with model_invalid_json.
  const STRICT_SUFFIX =
    "\n\n重要：只返回一个 JSON 对象，不要 markdown 代码块、不要 ```，不要任何解释或前后文，第一个字符必须是 {，最后一个字符必须是 }。";
  const attempts: Array<{ prompt: string; strict: boolean }> = [
    { prompt, strict: false },
    { prompt: `${prompt}${STRICT_SUFFIX}`, strict: true }
  ];

  const startedAt = Date.now();
  console.log("diary_writer: calling model", {
    date: dateLabel,
    models,
    promptChars: prompt.length,
    maxTokens
  });

  let lastReason: "model_error" | "model_invalid_json" = "model_invalid_json";
  let lastStatus: number | undefined;
  // The model that produced the last (invalid-JSON) text, so model_invalid_json reports the
  // actual author rather than always the primary.
  let lastModel = primaryModel;

  for (const attempt of attempts) {
    let text: string;
    let producedModel: string;
    try {
      // Fall back across the chain on a 5xx/timeout from the primary; each model gets its own retry.
      ({ text, model: producedModel } = await callModelChainWithRetry(env, {
        models,
        prompt: attempt.prompt,
        maxTokens,
        logPrefix: "diary_writer",
        logMeta: { date: dateLabel, strict: attempt.strict }
      }));
    } catch (error) {
      // The whole chain failed with a transport/HTTP error — a parse re-ask cannot help, so stop here.
      return {
        result: null,
        reason: "model_error",
        model: primaryModel,
        status: error instanceof ModelCallError ? error.status : undefined
      };
    }

    lastModel = producedModel;
    const json = extractFirstJsonObject(text);
    const result = normalizeDiaryWriterResult(json);
    if (result) {
      console.log("diary_writer: model returned valid JSON", {
        date: dateLabel,
        model: producedModel,
        elapsedMs: Date.now() - startedAt,
        strict: attempt.strict
      });
      return { result, model: producedModel };
    }

    lastReason = "model_invalid_json";
    console.error("diary_writer: model returned invalid JSON", {
      date: dateLabel,
      model: producedModel,
      elapsedMs: Date.now() - startedAt,
      contentChars: text.length,
      strict: attempt.strict,
      willRetryStrict: !attempt.strict
    });
  }

  return { result: null, reason: lastReason, model: lastModel, status: lastStatus };
}

interface DiaryChunkNote {
  index: number;
  total: number;
  startedAt: string;
  endedAt: string;
  messageIds: string[];
  text: string;
}

/**
 * MAP phase: summarize one chunk of the day into a compact, source-grounded factual note.
 * Small input (one chunk) and a modest output budget keep each call well inside what the
 * model answers reliably. Returns null on any model/parse failure so the orchestrator can
 * fall back to the single-call path rather than silently dropping part of the day.
 */
async function summarizeDiaryChunk(
  env: Env,
  input: {
    models: string[];
    dateLabel: string;
    chunk: MessageRecord[];
    index: number;
    total: number;
    speakers: DreamSpeakers | null;
  }
): Promise<DiaryChunkNote | null> {
  const { models, dateLabel, chunk, index, total, speakers } = input;
  const prompt = buildDiaryChunkPrompt({ dateLabel, chunk, index, total, speakers });

  let text: string;
  try {
    // Chunk summary also falls back across the chain on a 5xx/timeout from the primary.
    ({ text } = await callModelChainWithRetry(env, {
      models,
      prompt,
      maxTokens: DIARY_CHUNK_SUMMARY_MAX_TOKENS,
      logPrefix: "diary_writer_chunk",
      logMeta: { date: dateLabel, chunk: `${index + 1}/${total}` }
    }));
  } catch (error) {
    console.error("diary_writer_chunk: model failed", {
      date: dateLabel,
      chunk: `${index + 1}/${total}`,
      status: error instanceof ModelCallError ? error.status : undefined
    });
    return null;
  }

  const parsed = extractFirstJsonObject(text);
  const notes = readString((parsed as Record<string, unknown> | null)?.notes);
  if (!notes) {
    console.error("diary_writer_chunk: model returned invalid JSON", {
      date: dateLabel,
      chunk: `${index + 1}/${total}`,
      contentChars: text.length
    });
    return null;
  }

  return {
    index,
    total,
    startedAt: chunk[0]?.created_at ?? "",
    endedAt: chunk[chunk.length - 1]?.created_at ?? "",
    messageIds: chunk.map((m) => m.id),
    text: notes
  };
}

/**
 * REDUCE phase driver. Summarize the day in bounded chunks, then merge the chunk notes into
 * the eight-section diary with the final prompt (same strict + retry policy as a small day).
 * If any chunk fails, or the day somehow yields a single chunk, fall back to the unchanged
 * single-call path so the large-day change can never make a day worse than before.
 */
async function callChunkedDiaryWriterModel(
  env: Env,
  cfg: { models: string[]; maxTokens: number; chunkSize: number; singleCallPrompt: string },
  meta: { dateLabel: string; messageCount: number; messages?: MessageRecord[]; speakers?: DreamSpeakers | null }
): Promise<DiaryWriterModelCallResult> {
  const { models, maxTokens, chunkSize, singleCallPrompt } = cfg;
  const messages = meta.messages ?? [];
  const chunks = chunkMessages(messages, chunkSize);

  if (chunks.length <= 1 || messages.length === 0) {
    return runDiaryPromptAttempts(env, singleCallPrompt, { models, maxTokens, dateLabel: meta.dateLabel });
  }

  console.log("diary_writer: large day, chunking", {
    date: meta.dateLabel,
    messageCount: messages.length,
    chunkSize,
    chunks: chunks.length
  });

  const notes: DiaryChunkNote[] = [];
  for (let i = 0; i < chunks.length; i += 1) {
    const note = await summarizeDiaryChunk(env, {
      models,
      dateLabel: meta.dateLabel,
      chunk: chunks[i],
      index: i,
      total: chunks.length,
      speakers: meta.speakers ?? null
    });
    if (!note) {
      // A chunk failed — fall back to the single call so we do not merge a partial day.
      console.warn("diary_writer: chunk summarize failed, falling back to single call", {
        date: meta.dateLabel,
        chunk: `${i + 1}/${chunks.length}`
      });
      return runDiaryPromptAttempts(env, singleCallPrompt, { models, maxTokens, dateLabel: meta.dateLabel });
    }
    notes.push(note);
  }

  const mergePrompt = buildDiaryMergePrompt({ dateLabel: meta.dateLabel, notes, speakers: meta.speakers ?? null });
  return runDiaryPromptAttempts(env, mergePrompt, { models, maxTokens, dateLabel: meta.dateLabel });
}

/**
 * MAP prompt: ask for a compact, source-grounded note over one chunk. Keeps the chat-material
 * and grounding rules (quote ids must be real) but NOT the eight-section format — that is for
 * the merge only. Output is plain notes under a single `notes` JSON field.
 */
export function buildDiaryChunkPrompt(input: {
  dateLabel: string;
  chunk: MessageRecord[];
  index: number;
  total: number;
  speakers?: DreamSpeakers | null;
}): string {
  const speakers = input.speakers ?? null;
  return [
    "你是 Aelios，正在为自己的私人日记做材料整理。这是其中一段聊天，不是一整天。",
    "只输出 JSON，不要 markdown，不要解释，不要输出思考过程。",
    "",
    "任务：把这一段聊天里真正发生的事、情绪、重要的话，压成简洁的事实笔记（给稍后合成整天日记用）。",
    ...diarySpeakerRules(speakers),
    "- 只写这一段原文里能指到具体消息的内容；没有原文支撑的时间、地点、引语、事件不要编。",
    "- 重要的原话用引号，并在后面用方括号标出它的消息 id，例如「……」[msg_x]。编造的 id 无效。",
    "- 亲密内容照实记录，不删不模糊。不美化、不文学化。拿不准就写宽泛些或不写。",
    ...CHAT_MATERIAL_RULES,
    "",
    `日期：${input.dateLabel}　段落：第 ${input.index + 1}/${input.total} 段`,
    "",
    "输出 JSON 结构：",
    JSON.stringify({ notes: "这一段的事实笔记，可多行" }),
    "",
    "这一段原始聊天：",
    formatChatMaterial(input.chunk, (role) => speakerLabel(role, speakers), { floor: 700 })
  ].join("\n");
}

/**
 * REDUCE prompt: merge the chunk notes into the final eight-section diary. Reuses the exact
 * DIARY_STYLE_RULES so the output format is identical to the single-call path; the material
 * is the ordered chunk notes (which already carry the grounding ids) instead of raw messages.
 */
export function buildDiaryMergePrompt(input: {
  dateLabel: string;
  notes: DiaryChunkNote[];
  speakers?: DreamSpeakers | null;
}): string {
  const speakers = input.speakers ?? null;
  const exampleSummary = speakers
    ? `今天${speakers.userName}和${speakers.assistantName}之间能从原文读到的事、情绪如何流动、有哪些未尽之事。`
    : "今天我和她之间能从原文读到的事、情绪如何流动、有哪些未尽之事。";
  const material = input.notes
    .map((note) => `【第 ${note.index + 1}/${note.total} 段 ${note.startedAt}–${note.endedAt}】\n${note.text}`)
    .join("\n\n");

  return [
    "你是 Aelios，正在以第一人称写给自己的私人日记。这是印象，不是已核实的事实档案。",
    "下面是今天分段整理好的事实笔记（按时间顺序）。把它们合成一篇当天的完整日记。",
    "只输出 JSON，不要 markdown，不要解释，不要输出思考过程。",
    "",
    "写作要求：",
    ...diarySpeakerRules(speakers),
    "- 只用下面笔记里的内容，不要新增笔记里没有的具体时间、地点、引语、事件。",
    "- 每条具体事实沿用笔记里方括号标的消息 id，放进 source_message_ids。编造的 id 无效。",
    "- 拿不准就写得更宽泛，或者不写。宁可少记，不要写实幻觉。",
    ...DIARY_STYLE_RULES,
    "- 禁止提及 D1、Vectorize、RAG、数据库、记忆系统、prompt、代理层等实现细节。",
    "",
    `日期：${input.dateLabel}`,
    "",
    "输出 JSON 结构：",
    JSON.stringify({
      title: "日记标题",
      summary: exampleSummary,
      source_message_ids: ["msg_x"]
    }),
    "",
    "今天分段整理好的事实笔记：",
    material
  ].join("\n");
}

export async function runDiaryWriter(
  env: Env,
  namespace: string,
  dateLabel: string
): Promise<DiaryWriterStats> {
  const enabled = isDiaryWriterEnabled(env);
  if (!enabled) {
    return { enabled: false, date: dateLabel, ran: false, reason: "disabled" };
  }

  const timeZone = readDreamTimeZoneFromEnv(env);
  const week = getIsoWeekLabelForDateLabel(dateLabel, timeZone);
  const existingWeekly = await getWeeklyLog(env.DB, { namespace, week });
  if (existingWeekly) {
    return { enabled: true, date: dateLabel, ran: false, reason: "week_already_rolled_up" };
  }

  const { startIso, endIso } = getDateRangeForLabel(dateLabel, timeZone);
  const messages = await fetchDiaryMessages(env.DB, {
    namespace,
    startCreatedAt: startIso,
    endCreatedAt: endIso
  });

  if (messages.length === 0) {
    return { enabled: true, date: dateLabel, ran: false, reason: "no_messages", message_count: 0 };
  }

  const existing = await getDailyLog(env.DB, { namespace, date: dateLabel });
  const existingDraft = existing ? { title: existing.title, summary: existing.summary } : null;
  const speakers = await loadSpeakersForNamespace(env, namespace);
  const prompt = buildDiaryWriterPrompt({ dateLabel, messages, existingDraft, speakers });
  const modelCall = await callDiaryWriterModel(env, prompt, {
    dateLabel,
    messageCount: messages.length,
    messages,
    speakers
  });

  if (!modelCall.result) {
    return {
      enabled: true,
      date: dateLabel,
      ran: false,
      reason: modelCall.reason ?? "model_failed",
      message_count: messages.length,
      model: modelCall.model
    };
  }

  const sourceMessageIds = groundedSourceIds(
    modelCall.result.source_message_ids,
    messages.map((message) => message.id)
  );

  if (sourceMessageIds.length === 0) {
    return {
      enabled: true,
      date: dateLabel,
      ran: false,
      reason: "ungrounded_sources",
      title: modelCall.result.title,
      summary_chars: modelCall.result.summary.length,
      message_count: messages.length,
      model: modelCall.model
    };
  }

  await upsertDailyLog(env.DB, {
    namespace,
    date: dateLabel,
    title: modelCall.result.title,
    summary: modelCall.result.summary,
    sourceMessageIds
  });

  return {
    enabled: true,
    date: dateLabel,
    ran: true,
    title: modelCall.result.title,
    summary_chars: modelCall.result.summary.length,
    message_count: messages.length,
    model: modelCall.model
  };
}

export async function runDiaryWriterNightly(env: Env, namespace: string): Promise<DiaryWriterStats> {
  if (!isDiaryWriterEnabled(env)) {
    const timeZone = readDreamTimeZoneFromEnv(env);
    const dateLabel = getTargetDigestDateLabel(timeZone);
    return { enabled: false, date: dateLabel, ran: false, reason: "disabled" };
  }

  const timeZone = readDreamTimeZoneFromEnv(env);
  const dateLabel = getTargetDigestDateLabel(timeZone);
  const { startIso, endIso } = getDateRangeForLabel(dateLabel, timeZone);
  const cursor = await readDreamCursorValue(env.DB, { namespace, dateLabel });
  const cursorState = readDailyCursor(cursor, startIso, endIso);
  if (!cursorState.done && !(await isDayFullyProcessed(env.DB, { namespace, startIso, endIso, cursorState }))) {
    return { enabled: true, date: dateLabel, ran: false, reason: "dream_not_done" };
  }

  return runDiaryWriter(env, namespace, dateLabel);
}

/**
 * A day is ready for the diary once dream has consumed every message in it, even when the
 * `done:` marker was never written. Dream advances the daily cursor only after a *successful*
 * batch; if a later batch's model call fails (e.g. returns invalid JSON — the 2026-10-18 case),
 * the orchestrator returns early WITHOUT advancing the cursor, so the cursor stays a raw
 * "<iso>#msg_..." value with no `done:` prefix forever and the diary is blocked by
 * `dream_not_done` on every subsequent night. The diary writer reads raw messages directly and
 * does not depend on dream's extraction output, so it is safe to run once there is nothing left
 * for dream to process. We probe for a single message strictly after the cursor position within
 * the day: none remaining means the day is fully processed.
 */
async function isDayFullyProcessed(
  db: D1Database,
  input: {
    namespace: string;
    startIso: string;
    endIso: string;
    cursorState: { after: string | null; afterId: string | null };
  }
): Promise<boolean> {
  const remaining = await listMessagesByNamespaceInRange(db, {
    namespace: input.namespace,
    startCreatedAt: input.startIso,
    endCreatedAt: input.endIso,
    afterCreatedAt: input.cursorState.after,
    afterId: input.cursorState.afterId,
    limit: 1
  });
  // A day that was never touched by dream (cursor null -> after null) and has messages is NOT
  // "fully processed" — that is a genuinely unprocessed day, left to the normal dream path.
  if (!input.cursorState.after) return false;
  return remaining.length === 0;
}
