import { getMemoryById } from "../db/memories";
import type { Env, MemoryApiRecord } from "../types";
import { readNumber } from "../utils/request";
import { searchVectorMemories } from "./vectorStore";
import { intimateDedupCosine, isIntimateLayer } from "./layers";

export interface SimilarHit {
  memory: MemoryApiRecord;
  score: number;
}

export async function findSimilarActiveMemory(
  env: Env,
  input: { namespace: string; content: string; excludeIds?: string[]; tags?: readonly string[] }
): Promise<SimilarHit | null> {
  try {
    const baseThreshold = readNumber(env.DEDUP_COSINE, 0.9);
    // 1d 亲密去重更严：候选任一侧是「亲密」层时，用更高的阈值 (默认 0.97)。
    //   不同亲密夜晚用词高度雷同，普通 0.9 会把它们误判成重复合并掉；只对亲密侧收紧，其它层不变。
    const intimateThreshold = Math.max(baseThreshold, intimateDedupCosine(env));
    const newSideIntimate = isIntimateLayer(input.tags);
    const exclude = new Set(input.excludeIds ?? []);
    const hits = await searchVectorMemories(env, {
      namespace: input.namespace,
      query: input.content,
      topK: 5
    });

    let best: SimilarHit | null = null;
    for (const memory of hits) {
      if (exclude.has(memory.id)) continue;
      const score = memory.score ?? 0;
      const d1Row = await getMemoryById(env.DB, { namespace: input.namespace, id: memory.id });
      if (!d1Row || d1Row.status !== "active" || d1Row.version_status === "superseded") continue;
      // 任一侧 (新候选 或 已有记忆) 是亲密层 → 用更严阈值。
      const oldSideIntimate = isIntimateLayer(parseRowTags(d1Row.tags));
      const threshold = newSideIntimate || oldSideIntimate ? intimateThreshold : baseThreshold;
      if (score < threshold) continue;
      if (!best || score > best.score) {
        best = { memory, score };
      }
    }
    return best;
  } catch (error) {
    console.warn("dedup_gate: search failed (fail-open)", {
      reason: error instanceof Error ? error.message : String(error)
    });
    return null;
  }
}

function parseRowTags(raw: string | null): string[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw) as unknown;
    return Array.isArray(parsed) ? parsed.filter((t): t is string => typeof t === "string") : [];
  } catch {
    return [];
  }
}