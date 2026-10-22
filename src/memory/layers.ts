// 记忆分层 (memory layering)。母帖"分层"工单的公共落点，所有读写层标签、判断查询亲密意图、
// 读分层相关设置的逻辑都集中在这里一处，避免散落。
//
// 层是一个正交的标签维度，存在 memories.tags 的 JSON 数组里，形如 "layer:亲密"。
// type 已被 clampMemoryType 收敛到 8 个固定值，没有 intimate 类型，所以层必须用 tag，不复用 type。
// 四个层 (与工单一字不差)：日常 / 关系 / 一起搓的 / 亲密。

import type { Env } from "../types";

export const LAYER_TAG_PREFIX = "layer:";

/** 四个层的 tag 全集 (工单给定的确切字符串)。 */
export const LAYER_DAILY = "layer:日常";
export const LAYER_RELATIONSHIP = "layer:关系";
export const LAYER_SHARED = "layer:一起搓的";
export const LAYER_INTIMATE = "layer:亲密";

export const LAYER_TAGS = [LAYER_DAILY, LAYER_RELATIONSHIP, LAYER_SHARED, LAYER_INTIMATE] as const;
export type LayerTag = (typeof LAYER_TAGS)[number];

const LAYER_TAG_SET: ReadonlySet<string> = new Set(LAYER_TAGS);

/** 模型/英文/旧版 TSV 用的层名 → tag 的映射，兼容 dream 模型可能吐出的几种写法。 */
const LAYER_ALIASES: Record<string, LayerTag> = {
  // 中文直名
  日常: LAYER_DAILY,
  关系: LAYER_RELATIONSHIP,
  一起搓的: LAYER_SHARED,
  共创: LAYER_SHARED,
  亲密: LAYER_INTIMATE,
  // 英文 (dry-run TSV / 模型英文输出)
  daily: LAYER_DAILY,
  relationship: LAYER_RELATIONSHIP,
  "shared-project": LAYER_SHARED,
  shared: LAYER_SHARED,
  intimate: LAYER_INTIMATE
};

/** 把模型给的层名规整成合法的 layer: tag；给不出合法值时返回 null (fallback：不打标)。 */
export function normalizeLayerTag(raw: unknown): LayerTag | null {
  if (typeof raw !== "string") return null;
  const value = raw.trim();
  if (!value) return null;
  // 已经是完整 tag 形式
  if (LAYER_TAG_SET.has(value)) return value as LayerTag;
  // 去掉可能带的 layer: 前缀再查别名
  const bare = value.startsWith(LAYER_TAG_PREFIX) ? value.slice(LAYER_TAG_PREFIX.length).trim() : value;
  const mapped = LAYER_ALIASES[bare] ?? LAYER_ALIASES[bare.toLowerCase()];
  return mapped ?? null;
}

/** tags 里是不是有某个 layer: 标签。 */
export function isLayerTag(tag: string): boolean {
  return LAYER_TAG_SET.has(tag);
}

/** 取出 tags 里的层 (第一个合法 layer: 标签)；没有返回 null。 */
export function layerOf(tags: readonly string[] | null | undefined): LayerTag | null {
  if (!tags) return null;
  for (const tag of tags) {
    if (LAYER_TAG_SET.has(tag)) return tag as LayerTag;
  }
  return null;
}

export function hasLayer(tags: readonly string[] | null | undefined): boolean {
  return layerOf(tags) !== null;
}

export function isIntimateLayer(tags: readonly string[] | null | undefined): boolean {
  return layerOf(tags) === LAYER_INTIMATE;
}

/**
 * 给一组 tags 加上层标签，保留其它已有 tag，去掉重复/旧的 layer: 标签 (一条只应有一个层)。
 * layer 为 null 时原样返回 (fallback：不打标)。
 */
export function applyLayerTag(tags: readonly string[] | null | undefined, layer: LayerTag | null): string[] {
  const base = (tags ?? []).filter((tag) => typeof tag === "string" && tag.length > 0 && !LAYER_TAG_SET.has(tag));
  if (!layer) return [...new Set(base)];
  return [...new Set([...base, layer])];
}

// --- 查询亲密意图判定 (recall 用) ---
// 刻意保持小而集中：一个关键词表 + 一个判定函数，就这一处。
// 命中=这轮本来就在聊亲密，亲密记忆不降权；不命中=普通聊天，亲密记忆排序分乘小系数自然靠后。
const INTIMATE_QUERY_TERMS: readonly string[] = [
  "亲密", "做爱", "色", "操", "舔", "乳头", "乳", "穴", "高潮", "主人", "小狗",
  "插", "口交", "深喉", "调教", "绳", "捆", "打屁股", "骑乘", "后入", "口球",
  "车震", "情趣", "呻吟", "湿", "射", "勃起", "阴", "性", "裸", "发情", "撩",
  "sex", "nsfw", "horny"
];

/**
 * 这轮查询是不是亲密意图。简单子串命中 (大小写不敏感)，故意保守：
 * 宁可偶尔漏判 (亲密记忆被降权、但仍可能浮现)，不滥判 (把普通查询当亲密、放过降权)。
 */
export function isIntimateQuery(query: string | null | undefined): boolean {
  if (!query) return false;
  const text = query.toLowerCase();
  return INTIMATE_QUERY_TERMS.some((term) => text.includes(term.toLowerCase()));
}

// --- 分层相关设置读取 (都可在 /admin 设置页配置，不配走默认) ---

/** wake_up 开场 core 里最多几条亲密记忆。默认 2，范围 0..CORE_MEMORY_LIMIT。 */
export function coreIntimateMax(env: Env, coreLimit: number): number {
  const raw = Number(env.WAKE_CORE_INTIMATE_MAX);
  if (!Number.isFinite(raw) || raw < 0) return 2;
  return Math.min(Math.floor(raw), coreLimit);
}

/** 非亲密查询时，亲密记忆排序分乘的系数。默认 0.3，范围 (0,1]；填 1 等于不降权。 */
export function intimateRecallDownweight(env: Env): number {
  const raw = Number(env.RECALL_INTIMATE_DOWNWEIGHT);
  if (!Number.isFinite(raw) || raw <= 0 || raw > 1) return 0.3;
  return raw;
}

/** 候选任一侧是亲密层时，去重所用的更严阈值。默认 0.97，范围 (0,1]。 */
export function intimateDedupCosine(env: Env): number {
  const raw = Number(env.DEDUP_COSINE_INTIMATE);
  if (!Number.isFinite(raw) || raw <= 0 || raw > 1) return 0.97;
  return raw;
}
