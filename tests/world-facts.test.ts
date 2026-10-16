import { strict as assert } from "node:assert";
import { test, beforeEach } from "node:test";
import { DatabaseSync } from "node:sqlite";
import { readFileSync, readdirSync } from "node:fs";
import { timingSafeEqual } from "node:crypto";
import worker from "../src/index";
import { buildDreamExtractPrompt } from "../src/memory/dreamExtract";
import { buildDigestPrompt } from "../src/memory/dream/extractPhase";

// 世界知识页按 fact_key 前缀列记忆（world: / world_fact:），不再按 type=world_fact——
// v3-slim 把 type 钳到 8 个固定值后，world_fact 根本写不进去，老的 type 过滤永远是空的。
(crypto.subtle as any).timingSafeEqual = (a: Uint8Array, b: Uint8Array) => timingSafeEqual(a, b);
let sqlite: DatabaseSync;
let env: any;

beforeEach(() => {
  sqlite?.close();
  sqlite = new DatabaseSync(":memory:");
  for (const file of readdirSync("migrations").filter((f: string) => f.endsWith(".sql")).sort()) {
    try {
      sqlite.exec(readFileSync(`migrations/${file}`, "utf8"));
    } catch (error) {
      if (!String(error).includes("fts5")) throw error;
    }
  }
  const db = {
    prepare(sql: string) {
      const statement = sqlite.prepare(sql);
      let args: any[] = [];
      const api = {
        bind(...values: any[]) {
          args = values;
          return api;
        },
        async first() {
          return statement.get(...args) || null;
        },
        async all() {
          return { results: statement.all(...args) };
        },
        async run() {
          const r = statement.run(...args);
          return { meta: { changes: r.changes } };
        }
      };
      return api;
    }
  };
  env = {
    DB: db,
    CHATBOX_API_KEY: "owner-key",
    MEMORY_MCP_API_KEY: "mcp-key",
    VECTORIZE: {
      async deleteByIds() {
        return { mutationId: "m" };
      },
      async upsert() {
        return { mutationId: "m" };
      }
    }
  };
});

function insertMemory(
  id: string,
  namespace: string,
  content: string,
  opts: { type?: string; factKey?: string | null } = {}
): void {
  const type = opts.type ?? "fact";
  const factKey = opts.factKey ?? null;
  sqlite
    .prepare(
      `INSERT INTO memories (id, namespace, type, content, status, pinned, tags, source_message_ids, vector_id,
        fact_key, version_status, created_at, updated_at)
       VALUES (?, ?, ?, ?, 'active', 0, '[]', '[]', ?, ?, 'current', '2026-09-22T00:00:00Z', '2026-09-22T00:00:00Z')`
    )
    .run(id, namespace, type, content, `mem_${id}`, factKey);
}

function call(path: string, key = "owner-key", init: RequestInit = {}) {
  return worker.fetch(
    new Request(`https://aelios.test${path}`, {
      ...init,
      headers: { authorization: `Bearer ${key}` }
    }),
    env,
    { waitUntil() {} } as any
  );
}

test("fact_key_prefix=world: lists both world: and world_fact: records (any type), skipping the rest", async () => {
  insertMemory("m_world1", "default", "GPT-5 已发布", { type: "fact", factKey: "world:gpt-5-released" });
  insertMemory("m_world2", "default", "旧写法世界事实", { type: "note", factKey: "world_fact:claude-price-cut" });
  insertMemory("m_plain", "default", "普通事实", { type: "fact", factKey: "preference:answer-style" });
  insertMemory("m_nokey", "default", "没有 fact_key 的事实", { type: "fact", factKey: null });
  // world: 前缀的近义但不同的 key 不应被钓进来
  insertMemory("m_worldly", "default", "不是世界事实", { type: "fact", factKey: "worldly:thing" });

  const res = await (await call("/v1/memory?status=active&limit=80&fact_key_prefix=world:")).json() as any;
  const keys = res.data.map((m: any) => m.fact_key).sort();
  assert.deepEqual(keys, ["world:gpt-5-released", "world_fact:claude-price-cut"]);
  // fact_key 一路透出到 API 记录，面板的编辑/删除/supersede 按 id 操作仍然成立
  for (const m of res.data) assert.ok(typeof m.id === "string" && m.id.length > 0);
});

test("fact_key_prefix matches world_fact: directly too", async () => {
  insertMemory("m_wf", "default", "直接 world_fact 前缀", { factKey: "world_fact:service-down" });
  insertMemory("m_other", "default", "别的", { factKey: "decision:x" });

  const res = await (await call("/v1/memory?status=active&limit=80&fact_key_prefix=world_fact:")).json() as any;
  assert.deepEqual(
    res.data.map((m: any) => m.fact_key),
    ["world_fact:service-down"]
  );
});

test("fact_key_prefix respects namespace scoping and auth (narrow keys stay pinned)", async () => {
  insertMemory("m_def", "default", "旦九的世界事实", { factKey: "world:a" });
  insertMemory("m_ws", "wangshu", "望舒的世界事实", { factKey: "world:b" });

  // owner key honours ?namespace=
  const ws = await (await call("/v1/memory?status=active&fact_key_prefix=world:&namespace=wangshu")).json() as any;
  assert.deepEqual(ws.data.map((m: any) => m.content), ["望舒的世界事实"]);

  // narrower mcp key cannot cross into wangshu — stays pinned to default
  const pinned = await (await call(
    "/v1/memory?status=active&fact_key_prefix=world:&namespace=wangshu",
    "mcp-key"
  )).json() as any;
  assert.deepEqual(pinned.data.map((m: any) => m.content), ["旦九的世界事实"]);

  // missing auth is rejected
  const noauth = await worker.fetch(
    new Request("https://aelios.test/v1/memory?status=active&fact_key_prefix=world:"),
    env,
    { waitUntil() {} } as any
  );
  assert.equal(noauth.status, 401);
});

test("the world: fact_key rule is present in the dream extract prompt", () => {
  const prompt = buildDreamExtractPrompt([]);
  assert.match(prompt, /fact_key 要以 world: 开头/);
  assert.match(prompt, /type=fact/);
  // 仍然保留 8 类型钳制，不新增 world_fact 类型
  assert.match(prompt, /绝不输出 project、world_fact/);
});

test("the digest update prompt also carries the world: rule", () => {
  const prompt = buildDigestPrompt({
    dateLabel: "2026-10-16",
    startIso: "2026-10-16T00:00:00.000Z",
    endIso: "2026-10-17T00:00:00.000Z",
    messages: [],
    existingMemories: [],
    hasMore: false
  });
  assert.match(prompt, /fact_key 设为 world:/);
});
