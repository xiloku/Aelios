import { strict as assert } from "node:assert";
import { test } from "node:test";
import {
  buildJudgeModelChain,
  DEFAULT_CANDIDATE_JUDGE_FALLBACK,
  DEFAULT_CANDIDATE_JUDGE_MODEL,
  isCompatJudgeModel,
  parseVerdict,
  resolveJudgeModel
} from "../src/memory/compatJudge";

test("candidate judge model defaults to deepseek-flash, respects the setting", () => {
  assert.equal(resolveJudgeModel({} as any), DEFAULT_CANDIDATE_JUDGE_MODEL);
  assert.equal(resolveJudgeModel({ CANDIDATE_JUDGE_MODEL: "" } as any), DEFAULT_CANDIDATE_JUDGE_MODEL);
  assert.equal(resolveJudgeModel({ CANDIDATE_JUDGE_MODEL: "deepseek/deepseek-v4-pro" } as any), "deepseek/deepseek-v4-pro");
});

test("non-clef models route through the compat path; the Workers-AI clef model does not", () => {
  assert.equal(isCompatJudgeModel("deepseek/deepseek-flash"), true);
  assert.equal(isCompatJudgeModel("@cf/cloudflare/clef"), false);
  assert.equal(isCompatJudgeModel("@CF/CloudFlare/CLEF"), false);
  assert.equal(isCompatJudgeModel(""), false);
});

test("the judge model chain always ends with the deepseek-v4-pro fallback", () => {
  const chain = buildJudgeModelChain({} as any);
  assert.equal(chain[0], DEFAULT_CANDIDATE_JUDGE_MODEL);
  assert.ok(chain.includes(DEFAULT_CANDIDATE_JUDGE_FALLBACK));
  // no duplicate fallback when it is already the primary
  const chain2 = buildJudgeModelChain({ CANDIDATE_JUDGE_MODEL: DEFAULT_CANDIDATE_JUDGE_FALLBACK } as any);
  assert.equal(chain2.filter((m) => m === DEFAULT_CANDIDATE_JUDGE_FALLBACK).length, 1);
});

test("parseVerdict maps add/update grounded+worth to the clef-shaped result", () => {
  const add = parseVerdict("add", '{"grounded": 0.9, "worth": 0.8}');
  assert.ok(add);
  assert.equal(add!.grounded, true);
  assert.equal(add!.durable, true);
  assert.equal(add!.score, 0.8);
  assert.equal(add!.shouldDelete, null);

  const weak = parseVerdict("add", '{"grounded": 0.2, "worth": 0.1}');
  assert.equal(weak!.grounded, false);
  assert.equal(weak!.durable, false);
});

test("parseVerdict maps delete archive to shouldDelete", () => {
  const del = parseVerdict("delete", '{"archive": 0.8}');
  assert.equal(del!.shouldDelete, true);
  assert.equal(del!.score, 0.8);
  const keep = parseVerdict("delete", '{"archive": 0.2}');
  assert.equal(keep!.shouldDelete, false);
});

test("parseVerdict tolerates surrounding prose/fences and rejects missing fields", () => {
  const fenced = parseVerdict("add", 'sure:\n```json\n{"grounded":0.7,"worth":0.6}\n```');
  assert.ok(fenced);
  assert.equal(fenced!.score, 0.6);
  assert.equal(parseVerdict("add", '{"grounded": 0.7}'), null); // worth missing
  assert.equal(parseVerdict("add", "not json"), null);
  assert.equal(parseVerdict("delete", '{"foo": 1}'), null);
});

test("parseVerdict clamps out-of-range probabilities into [0,1]", () => {
  const over = parseVerdict("delete", '{"archive": 1.7}');
  assert.equal(over!.score, 1);
  const under = parseVerdict("add", '{"grounded": -3, "worth": 0.5}');
  assert.equal(under!.grounded, false);
});
