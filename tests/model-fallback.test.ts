import { strict as assert } from "node:assert";
import { afterEach, test } from "node:test";
import {
  buildModelChain,
  callModelChainWithRetry,
  DEFAULT_MODEL_FALLBACKS,
  ModelCallError,
  readModelFallbacks
} from "../src/utils/modelCall";
import type { Env } from "../src/types";

/**
 * The diary writer and dream (nightly extraction) model calls share one fallback primitive:
 * callModelChainWithRetry tries an ordered chain of models for ONE logical call and advances to
 * the next model on a 5xx/timeout (ModelCallError), so a burst of HTTP 500s from the primary
 * (@cf/openai/gpt-oss-120b) no longer fails the whole night. These tests pin that behaviour:
 *   1. a 500 from the primary is retried with the next model (fallback TAKEN);
 *   2. a 200 from the primary never calls any fallback (fallback NOT taken);
 *   3. all models 500 -> the chain is exhausted and the final error propagates.
 * callOpenAICompat falls through to global fetch when no AI binding / gateway id is set, so we
 * stub fetch and route by the request body's `model`.
 */

const BASE_ENV = {
  AI_GATEWAY_BASE_URL: "https://gateway.example/v1/acct/gw"
} as unknown as Env;

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
});

function okBody(text: string): string {
  return JSON.stringify({ choices: [{ message: { content: text } }] });
}

/** Install a fetch stub that answers per-model and records the order models were tried. */
function stubFetchByModel(answers: Record<string, { status: number; text?: string }>): string[] {
  const tried: string[] = [];
  globalThis.fetch = (async (_url: string, init?: { body?: string }) => {
    const body = init?.body ? (JSON.parse(init.body) as { model: string }) : { model: "" };
    tried.push(body.model);
    const ans = answers[body.model] ?? { status: 500 };
    return new Response(ans.status === 200 ? okBody(ans.text ?? "{}") : JSON.stringify({ error: "boom" }), {
      status: ans.status,
      headers: { "content-type": "application/json" }
    });
  }) as unknown as typeof fetch;
  return tried;
}

test("readModelFallbacks returns the built-in default chain when unset", () => {
  assert.deepEqual(readModelFallbacks(BASE_ENV), DEFAULT_MODEL_FALLBACKS);
});

test("readModelFallbacks parses MODEL_FALLBACKS, trimming blanks and deduping", () => {
  const env = { ...BASE_ENV, MODEL_FALLBACKS: " a , b ,, a , c " } as unknown as Env;
  assert.deepEqual(readModelFallbacks(env), ["a", "b", "c"]);
});

test("buildModelChain puts the primary first and never duplicates it", () => {
  const env = { ...BASE_ENV, MODEL_FALLBACKS: "p,b,c" } as unknown as Env;
  assert.deepEqual(buildModelChain("p", env), ["p", "b", "c"]);
  assert.deepEqual(buildModelChain("x", env), ["x", "p", "b", "c"]);
});

test("fallback is TAKEN: a 500 from the primary retries the same call with the next model", async () => {
  const tried = stubFetchByModel({
    primary: { status: 500 },
    fallback1: { status: 200, text: '{"ok":true}' }
  });
  const { text, model } = await callModelChainWithRetry(BASE_ENV, {
    models: ["primary", "fallback1"],
    prompt: "hi",
    backoffMs: [] // single attempt per model keeps the test fast
  });
  assert.equal(model, "fallback1");
  assert.equal(text, '{"ok":true}');
  // primary tried first, then fallback1 after the 500.
  assert.deepEqual(tried, ["primary", "fallback1"]);
});

test("fallback is NOT taken: a 200 from the primary never calls a fallback", async () => {
  const tried = stubFetchByModel({
    primary: { status: 200, text: '{"done":1}' },
    fallback1: { status: 200, text: "SHOULD-NOT-RUN" }
  });
  const { text, model } = await callModelChainWithRetry(BASE_ENV, {
    models: ["primary", "fallback1"],
    prompt: "hi",
    backoffMs: []
  });
  assert.equal(model, "primary");
  assert.equal(text, '{"done":1}');
  assert.deepEqual(tried, ["primary"]);
});

test("chain exhausted: every model 500s -> ModelCallError propagates after trying all", async () => {
  const tried = stubFetchByModel({
    primary: { status: 500 },
    fallback1: { status: 500 },
    fallback2: { status: 500 }
  });
  await assert.rejects(
    () =>
      callModelChainWithRetry(BASE_ENV, {
        models: ["primary", "fallback1", "fallback2"],
        prompt: "hi",
        backoffMs: []
      }),
    (err: unknown) => {
      assert.ok(err instanceof ModelCallError);
      assert.equal((err as ModelCallError).status, 500);
      return true;
    }
  );
  assert.deepEqual(tried, ["primary", "fallback1", "fallback2"]);
});

test("single-element chain behaves like a plain call and reports the model", async () => {
  stubFetchByModel({ only: { status: 200, text: "{}" } });
  const { model } = await callModelChainWithRetry(BASE_ENV, {
    models: ["only"],
    prompt: "hi",
    backoffMs: []
  });
  assert.equal(model, "only");
});

/**
 * DeepSeek V4 runs thinking mode ON by default, which drains the whole token budget into
 * reasoning_content and leaves content empty (finish_reason:length -> model_invalid_json).
 * callModelWithRetry must send reasoning_effort:"none" for deepseek/* models to disable it,
 * and must NOT add that field for any other model (byte-identical request as before).
 */
function stubFetchCapturingBodies(status: number, text: string): Array<Record<string, unknown>> {
  const bodies: Array<Record<string, unknown>> = [];
  globalThis.fetch = (async (_url: string, init?: { body?: string }) => {
    bodies.push(init?.body ? (JSON.parse(init.body) as Record<string, unknown>) : {});
    return new Response(okBody(text), { status, headers: { "content-type": "application/json" } });
  }) as unknown as typeof fetch;
  return bodies;
}

test("deepseek/* request carries reasoning_effort:\"none\" (thinking disabled)", async () => {
  const bodies = stubFetchCapturingBodies(200, "{}");
  await callModelChainWithRetry(BASE_ENV, {
    models: ["deepseek/deepseek-flash"],
    prompt: "hi",
    backoffMs: []
  });
  assert.equal(bodies.length, 1);
  assert.equal(bodies[0].reasoning_effort, "none");
});

test("non-deepseek request does NOT carry reasoning_effort", async () => {
  const bodies = stubFetchCapturingBodies(200, "{}");
  await callModelChainWithRetry(BASE_ENV, {
    models: ["anthropic/claude-sonnet-4-5"],
    prompt: "hi",
    backoffMs: []
  });
  assert.equal(bodies.length, 1);
  assert.equal("reasoning_effort" in bodies[0], false);
});
