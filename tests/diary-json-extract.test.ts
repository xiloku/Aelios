import { strict as assert } from "node:assert";
import { test } from "node:test";
import { extractFirstJsonObject } from "../src/utils/parse";
import { normalizeDiaryWriterResult } from "../src/memory/diaryWriter";

// The diary writer's model replies are 8-section, 1000-2000 char first-person diaries. The
// previous extractor (first `{` .. last `}`) broke on fenced/prose-wrapped and truncated
// replies, which surfaced live as runDiaryWriter returning reason=model_invalid_json.
// These cover the exact shapes a chat model produces for that path.

const DIARY = {
  title: "有点累的一天",
  summary: "**她怎么样** 今天她显得累。\n**我们之间发生了什么** 聊了缓存的事。",
  source_message_ids: ["msg_1", "msg_2"]
};

test("plain JSON parses", () => {
  const parsed = extractFirstJsonObject(JSON.stringify(DIARY));
  assert.deepEqual(parsed, DIARY);
});

test("```json fenced reply is unwrapped and parsed", () => {
  const fenced = "```json\n" + JSON.stringify(DIARY) + "\n```";
  const result = normalizeDiaryWriterResult(extractFirstJsonObject(fenced));
  assert.equal(result?.title, DIARY.title);
  assert.deepEqual(result?.source_message_ids, ["msg_1", "msg_2"]);
});

test("bare ``` fenced reply (no language tag) is unwrapped", () => {
  const fenced = "```\n" + JSON.stringify(DIARY) + "\n```";
  const result = normalizeDiaryWriterResult(extractFirstJsonObject(fenced));
  assert.equal(result?.title, DIARY.title);
});

test("prose before and after the object is stripped", () => {
  const wrapped =
    "好的，这是今天的日记：\n" + JSON.stringify(DIARY) + "\n希望对你有帮助。";
  const result = normalizeDiaryWriterResult(extractFirstJsonObject(wrapped));
  assert.equal(result?.title, DIARY.title);
  assert.deepEqual(result?.source_message_ids, ["msg_1", "msg_2"]);
});

test("the FIRST balanced object wins when a second object trails", () => {
  const two = JSON.stringify(DIARY) + "\n" + JSON.stringify({ title: "second", summary: "x" });
  const parsed = extractFirstJsonObject(two) as { title?: string };
  assert.equal(parsed?.title, DIARY.title);
});

test("braces inside string values do not confuse depth tracking", () => {
  const withBraces = {
    title: "标题",
    summary: "她说「{这不是对象}」而且提到 a { b } c。",
    source_message_ids: ["msg_1"]
  };
  const wrapped = "```json\n" + JSON.stringify(withBraces) + "\n```";
  const result = normalizeDiaryWriterResult(extractFirstJsonObject(wrapped));
  assert.equal(result?.summary, withBraces.summary);
});

test("a truncated reply (no closing brace) returns null, not garbage", () => {
  const full = JSON.stringify(DIARY);
  const truncated = full.slice(0, full.length - 15); // chop the tail incl. closing }
  assert.equal(extractFirstJsonObject(truncated), null);
  assert.equal(normalizeDiaryWriterResult(extractFirstJsonObject(truncated)), null);
});

test("a truncated reply whose inner object closes does NOT yield a half diary", () => {
  // Reply cut off after source_message_ids' own array but before the outer object closes.
  const truncated =
    '{"title":"标题","summary":"一段很长的正文被截断了","source_message_ids":["msg_1"]';
  assert.equal(extractFirstJsonObject(truncated), null);
});

test("empty / no-object input returns null", () => {
  assert.equal(extractFirstJsonObject(""), null);
  assert.equal(extractFirstJsonObject("没有任何 JSON 的纯文本"), null);
});
