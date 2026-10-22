import { strict as assert } from "node:assert";
import { test } from "node:test";
import {
  applyLayerTag,
  coreIntimateMax,
  hasLayer,
  intimateDedupCosine,
  intimateRecallDownweight,
  isIntimateLayer,
  isIntimateQuery,
  LAYER_INTIMATE,
  LAYER_RELATIONSHIP,
  LAYER_SHARED,
  LAYER_DAILY,
  layerOf,
  normalizeLayerTag
} from "../src/memory/layers";
import { selectCoreWithLayerQuota } from "../src/api/mcp";
import { buildDreamExtractPrompt } from "../src/memory/dreamExtract";
import { buildDiaryWriterPrompt } from "../src/memory/diaryWriter";

// --- layers module: exact tag strings + normalization (1c store format) ---

test("the four layer tags are exactly the ticket strings", () => {
  assert.equal(LAYER_DAILY, "layer:日常");
  assert.equal(LAYER_RELATIONSHIP, "layer:关系");
  assert.equal(LAYER_SHARED, "layer:一起搓的");
  assert.equal(LAYER_INTIMATE, "layer:亲密");
});

test("normalizeLayerTag maps aliases, english, prefixed forms; junk -> null", () => {
  assert.equal(normalizeLayerTag("亲密"), LAYER_INTIMATE);
  assert.equal(normalizeLayerTag("intimate"), LAYER_INTIMATE);
  assert.equal(normalizeLayerTag("layer:亲密"), LAYER_INTIMATE);
  assert.equal(normalizeLayerTag("共创"), LAYER_SHARED); // dry-run TSV used 共创 for 一起搓的
  assert.equal(normalizeLayerTag("shared-project"), LAYER_SHARED);
  assert.equal(normalizeLayerTag("日常"), LAYER_DAILY);
  assert.equal(normalizeLayerTag("关系"), LAYER_RELATIONSHIP);
  assert.equal(normalizeLayerTag(""), null);
  assert.equal(normalizeLayerTag("不是层"), null);
  assert.equal(normalizeLayerTag(undefined), null);
  assert.equal(normalizeLayerTag(42), null);
});

test("applyLayerTag keeps other tags, enforces exactly one layer, is idempotent", () => {
  assert.deepEqual(applyLayerTag(["play", "健康"], LAYER_INTIMATE), ["play", "健康", "layer:亲密"]);
  // replaces an existing (possibly wrong) layer rather than keeping two
  assert.deepEqual(applyLayerTag(["layer:日常", "play"], LAYER_INTIMATE), ["play", "layer:亲密"]);
  // null layer = no tag (fallback), other tags preserved + de-duped
  assert.deepEqual(applyLayerTag(["play", "play"], null), ["play"]);
  // idempotent
  assert.deepEqual(applyLayerTag(["layer:关系"], LAYER_RELATIONSHIP), ["layer:关系"]);
});

test("layerOf / hasLayer / isIntimateLayer read the layer out of a tag array", () => {
  assert.equal(layerOf(["play", "layer:关系"]), LAYER_RELATIONSHIP);
  assert.equal(layerOf(["play"]), null);
  assert.equal(hasLayer(["layer:日常"]), true);
  assert.equal(hasLayer([]), false);
  assert.equal(isIntimateLayer(["layer:亲密"]), true);
  assert.equal(isIntimateLayer(["layer:关系"]), false);
  assert.equal(isIntimateLayer(null), false);
});

// --- 1b: query intimacy detection drives the downweight ---

test("isIntimateQuery matches the keyword list, case-insensitive, conservative", () => {
  assert.equal(isIntimateQuery("我们来做爱吧"), true);
  assert.equal(isIntimateQuery("主人"), true);
  assert.equal(isIntimateQuery("NSFW please"), true);
  assert.equal(isIntimateQuery("大闸蟹"), false); // the verification recall query must NOT trip intimacy
  assert.equal(isIntimateQuery("电话 语音"), false);
  assert.equal(isIntimateQuery(""), false);
  assert.equal(isIntimateQuery(null), false);
});

// --- settings getters: defaults + ranges ---

test("coreIntimateMax defaults to 2, clamps to [0, coreLimit]", () => {
  assert.equal(coreIntimateMax({} as any, 10), 2);
  assert.equal(coreIntimateMax({ WAKE_CORE_INTIMATE_MAX: "0" } as any, 10), 0);
  assert.equal(coreIntimateMax({ WAKE_CORE_INTIMATE_MAX: "4" } as any, 10), 4);
  assert.equal(coreIntimateMax({ WAKE_CORE_INTIMATE_MAX: "99" } as any, 10), 10);
  assert.equal(coreIntimateMax({ WAKE_CORE_INTIMATE_MAX: "-1" } as any, 10), 2);
  assert.equal(coreIntimateMax({ WAKE_CORE_INTIMATE_MAX: "abc" } as any, 10), 2);
});

test("intimateRecallDownweight defaults to 0.3, accepts (0,1], rejects junk", () => {
  assert.equal(intimateRecallDownweight({} as any), 0.3);
  assert.equal(intimateRecallDownweight({ RECALL_INTIMATE_DOWNWEIGHT: "0.5" } as any), 0.5);
  assert.equal(intimateRecallDownweight({ RECALL_INTIMATE_DOWNWEIGHT: "1" } as any), 1); // 1 = off
  assert.equal(intimateRecallDownweight({ RECALL_INTIMATE_DOWNWEIGHT: "0" } as any), 0.3);
  assert.equal(intimateRecallDownweight({ RECALL_INTIMATE_DOWNWEIGHT: "1.5" } as any), 0.3);
});

test("intimateDedupCosine defaults to 0.97, accepts (0,1]", () => {
  assert.equal(intimateDedupCosine({} as any), 0.97);
  assert.equal(intimateDedupCosine({ DEDUP_COSINE_INTIMATE: "0.95" } as any), 0.95);
  assert.equal(intimateDedupCosine({ DEDUP_COSINE_INTIMATE: "0" } as any), 0.97);
  assert.equal(intimateDedupCosine({ DEDUP_COSINE_INTIMATE: "2" } as any), 0.97);
});

// --- 1a: wake_up core layer quota ---

function rec(id: string, layer: string | null) {
  return { id, tags: layer ? JSON.stringify([layer]) : "[]" };
}

test("selectCoreWithLayerQuota caps intimate picks and backfills with next non-intimate", () => {
  // 4 intimate up front, then plenty of non-intimate; limit 10, cap 2
  //  -> 2 intimate + 8 non-intimate, the 2 extra intimate pushed out (dry-run SUMMARY_v2 behaviour).
  const records = [
    rec("i1", "layer:亲密"), rec("i2", "layer:亲密"), rec("i3", "layer:亲密"), rec("i4", "layer:亲密"),
    rec("n1", "layer:日常"), rec("n2", "layer:关系"), rec("n3", "layer:一起搓的"), rec("n4", "layer:日常"),
    rec("n5", "layer:关系"), rec("n6", "layer:一起搓的"), rec("n7", "layer:日常"), rec("n8", "layer:关系")
  ];
  const picked = selectCoreWithLayerQuota(records, 10, 2);
  assert.equal(picked.length, 10);
  const intimate = picked.filter((p) => isIntimateLayer(JSON.parse(p.tags)));
  assert.equal(intimate.length, 2);
  // the two kept intimate are the first two by original order
  assert.deepEqual(intimate.map((p) => p.id), ["i1", "i2"]);
  // the freed slots went to non-intimate by original order, intimate i3/i4 dropped out of the top 10
  assert.ok(!picked.some((p) => p.id === "i3" || p.id === "i4"));
});

test("selectCoreWithLayerQuota backfills intimate rather than returning fewer than the limit", () => {
  // cap is a ceiling, not an exclusion: with too few non-intimate it still returns a full 10,
  // backfilling with the pushed-out intimate records in original order. Needs >limit records so
  // the quota logic runs (at or below the limit everything is kept unchanged).
  const records = [
    rec("i1", "layer:亲密"), rec("i2", "layer:亲密"), rec("i3", "layer:亲密"),
    rec("i4", "layer:亲密"), rec("i5", "layer:亲密"), rec("i6", "layer:亲密"),
    rec("i7", "layer:亲密"), rec("i8", "layer:亲密"), rec("i9", "layer:亲密"),
    rec("n1", "layer:日常"), rec("n2", "layer:关系"), rec("n3", "layer:一起搓的")
  ];
  const picked = selectCoreWithLayerQuota(records, 10, 2);
  assert.equal(picked.length, 10);
  // first the 2 kept intimate + all 3 non-intimate, then intimate overflow backfills the rest
  assert.deepEqual(picked.map((p) => p.id).slice(0, 5), ["i1", "i2", "n1", "n2", "n3"]);
  assert.equal(picked.filter((p) => isIntimateLayer(JSON.parse(p.tags))).length, 7); // 2 kept + 5 backfilled fill the 10 slots
});

test("selectCoreWithLayerQuota is a no-op below the limit", () => {
  const records = [rec("a", "layer:亲密"), rec("b", "layer:亲密")];
  assert.deepEqual(selectCoreWithLayerQuota(records, 10, 2).map((r) => r.id), ["a", "b"]);
});

// --- 1c: dream extract prompt asks for a layer and shows the field ---

test("dream extract prompt instructs the model to output one of the four layers", () => {
  const prompt = buildDreamExtractPrompt([]);
  assert.match(prompt, /layer 字段/);
  assert.match(prompt, /日常、关系、一起搓的、亲密/);
  assert.match(prompt, /"layer":\s*"日常"/); // output example carries the field
  assert.match(prompt, /这条的 layer 固定填「亲密」/); // mandatory intimate record
});

// --- 1g: diary writing rules (遥 / 她 / chronological) ---

test("diary prompt carries the 遥 / 她 / chronological rules", () => {
  const prompt = buildDiaryWriterPrompt({ dateLabel: "2026-10-22", messages: [], existingDraft: null });
  assert.match(prompt, /默认把她叫「遥」/);
  assert.match(prompt, /不要用全名「潇遥」/);
  assert.match(prompt, /提到她一律用「她」，绝不用「他」/);
  assert.match(prompt, /事件按时间先后顺序叙述/);
});
