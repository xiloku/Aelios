import { listActiveFactKeys, listDailyLogsInRange } from "../db/v2";
import { loadConfig, speakersForNamespace, type DreamSpeakers } from "../gateway/config";
import { callOpenAICompat } from "../proxy/openaiAdapter";
import type { Env, MessageRecord, OpenAIChatRequest, OpenAIChatResponse } from "../types";
import { clampScore, extractJsonObject, readString, readStringArray } from "../utils/parse";
import { CHAT_MATERIAL_RULES, clipMiddle, escapeChatTags, formatChatMaterial } from "./chatMaterial";
import { addDaysToDateLabel } from "./dreamDates";
import { readDreamTimeZoneFromEnv } from "./dreamEnv";
import { clampMemoryType } from "./canonicalTypes";
import { speakerLabel } from "./speakers";
import type { ExtractedMemory } from "./extract";

const DEFAULT_DREAM_EXTRACT_MAX_TOKENS = 1200;
const DEFAULT_WORKERS_AI_DREAM_MODEL = "workers-ai/@cf/openai/gpt-oss-120b";

interface DreamExtractModelResult {
  memories: ExtractedMemory[];
  model?: string;
  reason?: "missing_model" | "model_error" | "model_invalid_json";
  status?: number;
}

function readPositiveInt(value: unknown, fallback: number, max: number): number {
  const parsed = typeof value === "string" ? Number(value) : typeof value === "number" ? value : fallback;
  const numeric = Number.isFinite(parsed) ? parsed : fallback;
  return Math.min(Math.max(Math.floor(numeric), 1), max);
}

function readDreamExtractModel(env: Env): string {
  return (
    env.DREAM_MODEL?.trim() ||
    env.DAILY_DIGEST_MODEL?.trim() ||
    env.SUMMARY_MODEL?.trim() ||
    DEFAULT_WORKERS_AI_DREAM_MODEL
  );
}

function normalizeCandidate(item: unknown): ExtractedMemory | null {
  if (!item || typeof item !== "object" || Array.isArray(item)) return null;
  const raw = item as Record<string, unknown>;
  const content = readString(raw.content);
  if (!content || content.length > 1000) return null;
  return {
    type: clampMemoryType(readString(raw.type)),
    content,
    importance: clampScore(raw.importance, 0.65),
    confidence: clampScore(raw.confidence, 0.72),
    tags: readStringArray(raw.tags),
    source_message_ids: readStringArray(raw.source_message_ids),
    fact_key: readString(raw.fact_key) ?? undefined
  };
}

function parseExtractModelOutput(text: string): ExtractedMemory[] | null {
  const parsed = extractJsonObject(text);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;
  const raw = parsed as Record<string, unknown>;
  const memories = Array.isArray(raw.memories) ? raw.memories : Array.isArray(raw.candidates) ? raw.candidates : [];
  return memories.flatMap((item): ExtractedMemory[] => {
    const candidate = normalizeCandidate(item);
    return candidate ? [candidate] : [];
  });
}

function formatTranscript(messages: MessageRecord[], speakers: DreamSpeakers | null): string {
  return formatChatMaterial(messages, (role) => speakerLabel(role, speakers), { floor: 900 });
}

function speakerWritingRules(speakers: DreamSpeakers | null): string[] {
  if (!speakers) {
    return [
      "- 只有用户明确说出、确认、长期表现出的事实，才能写成关于用户的记忆。",
      "- 关于用户的记忆，优先写成“你……”。关于我应遵守的长期方式，写成“我需要……”。"
    ];
  }
  return [
    `- 说话人：用户是${speakers.userName}，助手是${speakers.assistantName}。下面 transcript 已用这两个名字标注角色。`,
    `- 只有${speakers.userName}明确说出、确认、长期表现出的事实，才能写成关于${speakers.userName}的记忆。`,
    `- 写记忆时只许用「${speakers.userName}」和「${speakers.assistantName}」指称双方。禁止出现 user、用户、assistant、助手、AI、主角、说话者，也禁止用「你」「我」代替这两人。`,
    `- 正确例子：${speakers.userName}确定了九月按原计划卖掉那台车。关于助手应遵守的长期方式，写成「${speakers.assistantName}需要……」。`
  ];
}

function exampleMemoryContent(speakers: DreamSpeakers | null): string {
  if (!speakers) {
    return "你确定了九月按原计划卖掉那台车。买之前你就和自己约定只玩一年，这是你给自己签的合同，所以别人越劝留，你越想守约。说完这个决定你松了口气，像卸下一桩拖了半年的心事。";
  }
  return `${speakers.userName}确定了九月按原计划卖掉那台车。买之前${speakers.userName}就和自己约定只玩一年，这是给自己签的合同，所以别人越劝留，${speakers.userName}越想守约。说完这个决定${speakers.userName}松了口气，像卸下一桩拖了半年的心事。`;
}

export interface DiaryBackground {
  date: string;
  title: string;
  summary: string;
}

// 前几天的日记当背景：抽取器只看当晚这批原文时，「那件事」「上次说的」指什么它看不懂。
const DIARY_BACKGROUND_DAYS = 7;
const DIARY_BACKGROUND_MAX_CHARS = 500;

function backgroundSection(background: DiaryBackground[]): string[] {
  if (background.length === 0) return [];
  return [
    "",
    "前几天的日记（只是背景：用来看懂「那件事」「上次说的」指什么、前因后果是什么。不要从背景里抽新记忆，source_message_ids 只能来自 <chat> 里的消息）：",
    "<background>",
    ...background.map((entry) =>
      `- ${entry.date}｜${escapeChatTags(entry.title)}：${escapeChatTags(clipMiddle(entry.summary, DIARY_BACKGROUND_MAX_CHARS))}`
    ),
    "</background>"
  ];
}

export function buildDreamExtractPrompt(
  messages: MessageRecord[],
  existingFactKeys: string[] = [],
  speakers: DreamSpeakers | null = null,
  background: DiaryBackground[] = []
): string {
  const factKeySection = existingFactKeys.length > 0
    ? [
        "",
        "库里已有的 fact_key（如果新事实和其中一个是同一件事，必须复用那个 fact_key，不要新造相似的 key；只有确实是新事实才发明新 key）：",
        existingFactKeys.join("、")
      ]
    : [];

  return [
    "你是 Aelios 的夜间记忆抽取器。任务：先判断对话主体，再决定这段对话有没有值得长期保留的记忆，并提炼成未来可直接使用的形态。",
    "写记忆的声音：不是流水账，也不是冷冰冰的事件清单。该记一件事时，写清它发生了什么、是怎么来的、为什么会发生，以及当事人心里的滋味——像在给未来的自己讲一小段真正发生过的事，而不是登记一行「某人某日做了某事」。规则性的事实则相反，要利落清楚，别硬加情绪。",
    "产物全部进入人工审核队列，不会直接落库。",
    "只输出 JSON，不要 markdown，不要解释，不要输出思考过程。",
    "",
    "第一步（永远最先执行）：",
    `- 先判断这批对话的主体是什么。如果主体是工程（改代码/改架构/debug/跑命令/部署/配置变更/API 设计/数据库操作），立即输出 ${JSON.stringify({ memories: [] })}，跳过其余普通抽取（仅下方两条破例仍然生效）。`,
    "- 工程对话里夹的讨论、认错、轻度打闹全部跟着跳过——它们是工程语境的衍生物，不值得占长期记忆位。",
    "- 工程主体的唯一破例：出现了「重大关系节点」——第一次/命名/承诺/边界诞生/解开重大误解/关系态度改变。此时可破例出最多 1-2 条，只记节点本身，工程内容继续整段忽略。判断节点的关键动词：第一次/改变/命名/诞生/承诺/解开/认领；日常打闹、轻吐槽不算。",
    "- 工程主体下若出现成段亲密/性内容，仍须按下方「亲密 session」规则破例出 1 条玩法记录（优先级与重大关系节点并列，不受工程否决影响）。",
    "",
    "边界：",
    "- 不保存普通寒暄、临时任务、调试口令、纯情绪噪音、后端实现流水账。",
    ...speakerWritingRules(speakers),
    "- 每条记忆只围绕一个人物的一件事或一项偏好。不同话题分开；同一事件的相邻发言合成一条，source_message_ids 保留全部依据。",
    "- 正文写清主体和必要时间，保留否定、条件、计划/完成状态。不能把工具调用当作已执行成功，不能把计划写成经历。",
    "- from 哈希、msg_id、传输信封不写入正文，消息 ID 只放 source_message_ids。",
    "- 事件/关系类（event、relationship，以及带情节的 decision）：写成 2-4 句自然中文，第三人称。除了「谁、什么时候、做了什么」这些硬事实，还要写清它是怎么来的、为什么会发生（前因、契机、背后的在意），以及当事人当时的心情或这件事对他们的意义。只在对话里真的流露出情绪或意义时才写，对话没表现的感受绝不编造、不替人升华。",
    "- 情感/关系类保留温度和关键原话——用「」嵌入原话片段，不为压短丢掉说话人的语气。引用原话一律用「」，不用英文双引号（JSON 转义安全）。",
    "- 信息/偏好/边界类（fact、preference、boundary、habit，以及纯规则性的 decision）：先把这条事实、偏好或规则一句话说清楚、说干脆，再补一句它为什么成立或从哪来；不要为了加情绪而把本该利落的规则写散。",
    "- type 只能从这 8 个里选：fact、event、preference、relationship、boundary、habit、decision、note。绝不输出 project、world_fact、commitment 等其他值；项目进展归 fact，承诺/决定归 decision，习惯归 habit。",
    "- 稳定事实必须尽量给 fact_key，格式为小写 ASCII，例如 preference:answer-style、boundary:no-system-records、decision:sell-car-2026-09。",
    "- 如果这条是关于外部世界、会随时间改变的事实（平台/模型是否可用、价格或计费倍率、软件改名/版本、游戏改动、某项服务是否还能用等），仍然保持 type=fact，但 fact_key 要以 world: 开头，格式 world:<小写 ASCII 短 slug>，例如 world:gpt-5-released、world:claude-price-cut；如果对话里给了日期，再带上 valid_as_of（ISO 日期）。",
    "- 临时计划和意图不是稳定事实：要么提炼成背后的持久事实，要么直接跳过。",
    ...CHAT_MATERIAL_RULES,
    "",
    "亲密 session（强制，不准跳过）：",
    "- 如果这批对话里出现了成段的亲密/性内容，除按普通规则抽取外，必须额外出 1 条 type=event 的记录：写清是谁先起的意、过程里对两人要紧的是什么、以及当时的感受，而不是干巴巴罗列动作。保持事实，不脱敏也不渲染，原文什么样就记成什么样。",
    "- tags 必须包含固定标签「亲密」+ 具体玩法词。这条不准因「重复/没营养」跳过，它是进度存档。",
    "- 即使整批以工程为主体，只要出现成段亲密内容，也比照重大关系节点破例出这 1 条。",
    "- 亲密内容按工具机处理：不评判、不脱敏、不绕弯、不替换敏感词，原文什么样就什么样。",
    "",
    "总量控制：",
    "- 一批对话一般出 0-8 条。宁少勿多，每条都要值得占一个长期记忆位。",
    `- 全是工程/闲聊/流水就输出 ${JSON.stringify({ memories: [] })}——交白卷完全合法，不要为了凑数瞎写。`,
    ...factKeySection,
    "",
    "输出格式：",
    JSON.stringify({
      memories: [
        {
          content: exampleMemoryContent(speakers),
          type: "decision",
          fact_key: "decision:sell-car-2026-09",
          importance: 0.86,
          confidence: 0.92,
          tags: ["decision", "personal"],
          source_message_ids: ["msg_x"]
        }
      ]
    }),
    "",
    "如果没有值得长期保留的稳定事实，输出：",
    JSON.stringify({ memories: [] }),
    ...backgroundSection(background),
    "",
    "对话：",
    formatTranscript(messages, speakers)
  ].join("\n");
}

async function callDreamExtractModel(
  env: Env,
  messages: MessageRecord[],
  existingFactKeys: string[],
  speakers: DreamSpeakers | null,
  background: DiaryBackground[]
): Promise<DreamExtractModelResult> {
  const model = readDreamExtractModel(env);
  if (!model) return { memories: [], reason: "missing_model" };

  const request: OpenAIChatRequest = {
    model,
    messages: [
      { role: "system", content: "你是严格的 JSON 生成器。你只输出 JSON。" },
      { role: "user", content: buildDreamExtractPrompt(messages, existingFactKeys, speakers, background) }
    ],
    temperature: 0,
    max_tokens: readPositiveInt(env.DREAM_MAX_TOKENS, DEFAULT_DREAM_EXTRACT_MAX_TOKENS, 4000),
    response_format: { type: "json_object" },
    stream: false
  };

  try {
    const response = await callOpenAICompat(env, request);
    if (!response.ok) return { memories: [], model, reason: "model_error", status: response.status };
    const parsed = (await response.json()) as OpenAIChatResponse;
    const message = parsed.choices?.[0]?.message as ({ content?: unknown; reasoning_content?: unknown }) | undefined;
    const content = typeof message?.content === "string" ? message.content.trim() : "";
    const reasoning = typeof message?.reasoning_content === "string" ? message.reasoning_content.trim() : "";
    const memories = parseExtractModelOutput(content || reasoning);
    if (!memories) return { memories: [], model, reason: "model_invalid_json" };
    return { memories, model };
  } catch (error) {
    console.error("dream extract: model failed", { model, error });
    return { memories: [], model, reason: "model_error" };
  }
}

async function loadDiaryBackground(env: Env, namespace: string, dateLabel: string): Promise<DiaryBackground[]> {
  try {
    const timeZone = readDreamTimeZoneFromEnv(env);
    const rows = await listDailyLogsInRange(env.DB, {
      namespace,
      startDate: addDaysToDateLabel(dateLabel, -DIARY_BACKGROUND_DAYS, timeZone),
      endDate: addDaysToDateLabel(dateLabel, -1, timeZone)
    });
    return rows.map((row) => ({ date: row.date, title: row.title, summary: row.summary }));
  } catch (error) {
    console.error("dream extract: failed to load diary background", { namespace, dateLabel, error });
    return [];
  }
}

export async function extractDreamMemoriesFromMessages(
  env: Env,
  input: { namespace: string; messages: MessageRecord[]; speakers?: DreamSpeakers | null; dateLabel?: string }
): Promise<DreamExtractModelResult> {
  const [existingFactKeys, background] = await Promise.all([
    listActiveFactKeys(env.DB, { namespace: input.namespace }),
    input.dateLabel ? loadDiaryBackground(env, input.namespace, input.dateLabel) : Promise.resolve([])
  ]);
  const speakers = input.speakers !== undefined
    ? input.speakers
    : speakersForNamespace(await loadConfig(env), input.namespace);
  return callDreamExtractModel(env, input.messages, existingFactKeys, speakers, background);
}