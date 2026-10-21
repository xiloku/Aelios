import { strict as assert } from "node:assert";
import { test } from "node:test";
import { formatBootStable } from "../src/assembler/types";
import { buildDiaryChunkPrompt, buildDiaryMergePrompt, buildDiaryWriterPrompt, normalizeDiaryWriterResult } from "../src/memory/diaryWriter";
import { buildWeeklyRollupPrompt } from "../src/memory/weeklyRollup";
import { buildMonthlyRollupPrompt } from "../src/memory/monthlyRollup";
import { groundedSourceIds, IMPRESSION_DISCLAIMER } from "../src/memory/impression";

test("diary prompt forbids invented specifics and requires source ids", () => {
  const prompt = buildDiaryWriterPrompt({
    dateLabel: "2026-08-27",
    messages: [{
      id: "msg_1",
      conversation_id: "c1",
      namespace: "partner-a",
      role: "user",
      content: "今天有点累",
      source: "chat",
      created_at: "2026-08-27T12:00:00.000Z"
    }],
    existingDraft: null
  });
  assert.match(prompt, /source_message_ids/);
  assert.match(prompt, /没有原文支撑/);
  assert.match(prompt, /傍晚下班后抱怨/);
  assert.match(prompt, /用「我」指代助手自己/);
  assert.doesNotMatch(prompt, /具体细节优先于抽象概括/);
  assert.doesNotMatch(prompt, /古法PPT/);
});

test("named speakers replace 用户/助手 in the diary prompt", () => {
  const prompt = buildDiaryWriterPrompt({
    dateLabel: "2026-08-27",
    messages: [{
      id: "msg_1",
      conversation_id: "c1",
      namespace: "default",
      role: "user",
      content: "今天有点累",
      source: "chat",
      created_at: "2026-08-27T12:00:00.000Z"
    }],
    existingDraft: null,
    speakers: { userName: "小南", assistantName: "小北" }
  });
  assert.match(prompt, /用户是小南，助手是小北/);
  assert.match(prompt, /\[msg_1\].*\[小南\]/);
  assert.match(prompt, /小南今天显得累/);
  assert.doesNotMatch(prompt, /用「我」指代助手自己/);
});

test("claimed source ids that are not in the day's transcript are dropped", () => {
  assert.deepEqual(
    groundedSourceIds(["msg_1", "msg_fake", "msg_1"], ["msg_1", "msg_2"]),
    ["msg_1"]
  );
  assert.deepEqual(groundedSourceIds(["msg_fake"], ["msg_1"]), []);
});

test("diary JSON keeps title/summary and reads source_message_ids", () => {
  const parsed = normalizeDiaryWriterResult({
    title: "有点累",
    summary: "她今天显得累。",
    source_message_ids: ["msg_1", ""]
  });
  assert.equal(parsed?.title, "有点累");
  assert.deepEqual(parsed?.source_message_ids, ["msg_1"]);
});

test("named speakers replace 用户/助手 in weekly and monthly rollups", () => {
  const speakers = { userName: "小南", assistantName: "小北" };
  const weekly = buildWeeklyRollupPrompt({
    week: "2026-W36",
    startDate: "2026-08-31",
    endDate: "2026-09-06",
    dailyLogs: [{ date: "2026-09-01", title: "累", summary: "小南有点累。" }],
    speakers
  });
  assert.match(weekly, /用户是小南，助手是小北/);
  assert.doesNotMatch(weekly, /我=助手/);
  const monthly = buildMonthlyRollupPrompt({
    month: "2026-09",
    weeklyLogs: [{ week: "2026-W36", title: "一周", summary: "还好。" }],
    speakers
  });
  assert.match(monthly, /禁止出现 user、用户、assistant、助手/);
  assert.doesNotMatch(monthly, /关于用户用「你」/);
});

test("boot impressions carry the issue #35 disclaimer", () => {
  const text = formatBootStable({
    impressions: {
      daily: { label: "2026-08-27", title: "昨日", summary: "聊了缓存" },
      weekly: null,
      monthly: null,
      max_chars: 1000
    },
    precious: [],
    glossary: [],
    schema_version: "v3-1",
    cache_prefix_end: true
  });
  assert.match(text, new RegExp(IMPRESSION_DISCLAIMER));
  assert.match(text, /聊了缓存/);
});

const chunkMessages = [
  {
    id: "msg_a",
    conversation_id: "c1",
    namespace: "ain",
    role: "user" as const,
    content: "今天有点累",
    source: "chat",
    created_at: "2026-10-19T01:00:00.000Z"
  },
  {
    id: "msg_b",
    conversation_id: "c1",
    namespace: "ain",
    role: "assistant" as const,
    content: "我在这里",
    source: "chat",
    created_at: "2026-10-19T01:05:00.000Z"
  }
];

test("large-day chunk prompt grounds quotes on ids but omits the eight-section format", () => {
  const prompt = buildDiaryChunkPrompt({
    dateLabel: "2026-10-19",
    chunk: chunkMessages,
    index: 0,
    total: 3,
    speakers: null
  });
  // It is a MAP note, not the final diary: it carries grounding + chat-material rules…
  assert.match(prompt, /notes/);
  assert.match(prompt, /编造的 id 无效/);
  assert.match(prompt, /第 1\/3 段/);
  assert.match(prompt, /\[msg_a\]/);
  // …but NOT the eight-section diary structure (that belongs only to the merge).
  assert.doesNotMatch(prompt, /现在的温度/);
  assert.doesNotMatch(prompt, /source_message_ids/);
});

test("large-day merge prompt keeps the eight-section diary format and consumes chunk notes", () => {
  const prompt = buildDiaryMergePrompt({
    dateLabel: "2026-10-19",
    notes: [
      {
        index: 0,
        total: 2,
        startedAt: "2026-10-19T01:00:00.000Z",
        endedAt: "2026-10-19T01:30:00.000Z",
        messageIds: ["msg_a", "msg_b"],
        text: "她说累了[msg_a]。"
      },
      {
        index: 1,
        total: 2,
        startedAt: "2026-10-19T02:00:00.000Z",
        endedAt: "2026-10-19T02:30:00.000Z",
        messageIds: ["msg_c"],
        text: "后来一起安静待着[msg_c]。"
      }
    ],
    speakers: null
  });
  // The eight-section window-diary format is preserved verbatim from DIARY_STYLE_RULES.
  assert.match(prompt, /现在的温度/);
  assert.match(prompt, /未完成的事/);
  assert.match(prompt, /source_message_ids/);
  // It merges the ordered chunk notes (with their grounding ids), not raw transcript.
  assert.match(prompt, /第 1\/2 段/);
  assert.match(prompt, /第 2\/2 段/);
  assert.match(prompt, /\[msg_a\]/);
  assert.match(prompt, /\[msg_c\]/);
  // It must not invent material beyond the notes.
  assert.match(prompt, /不要新增笔记里没有的/);
});

test("named speakers flow into chunk and merge prompts", () => {
  const speakers = { userName: "小南", assistantName: "小北" };
  const chunk = buildDiaryChunkPrompt({ dateLabel: "2026-10-19", chunk: chunkMessages, index: 0, total: 2, speakers });
  assert.match(chunk, /用户是小南，助手是小北/);
  assert.match(chunk, /\[小南\]/);
  const merge = buildDiaryMergePrompt({
    dateLabel: "2026-10-19",
    notes: [
      { index: 0, total: 1, startedAt: "", endedAt: "", messageIds: ["msg_a"], text: "小南说累了[msg_a]。" }
    ],
    speakers
  });
  assert.match(merge, /用户是小南，助手是小北/);
});
