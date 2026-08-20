import assert from "node:assert/strict";
import test from "node:test";

import {
  LocalEngineRequestError,
  withRefreshedRequestToken,
} from "../app/local-engine.js";

test("uses the current request token without refreshing", async () => {
  const seen = [];
  const result = await withRefreshedRequestToken({
    token: "current-token",
    refreshToken: async () => {
      assert.fail("A valid token should not be refreshed.");
    },
    request: async (token) => {
      seen.push(token);
      return "ok";
    },
  });

  assert.equal(result, "ok");
  assert.deepEqual(seen, ["current-token"]);
});

test("fetches a token before the first mutation when none is cached", async () => {
  const seen = [];
  await withRefreshedRequestToken({
    token: "",
    refreshToken: async () => "fresh-token",
    request: async (token) => {
      seen.push(token);
    },
  });
  assert.deepEqual(seen, ["fresh-token"]);
});

test("refreshes and retries once after the engine rotates its token", async () => {
  const seen = [];
  let refreshes = 0;
  const result = await withRefreshedRequestToken({
    token: "old-token",
    refreshToken: async () => {
      refreshes += 1;
      return "new-token";
    },
    request: async (token) => {
      seen.push(token);
      if (token === "old-token") {
        throw new LocalEngineRequestError("invalid token", 403);
      }
      return "accepted";
    },
  });

  assert.equal(result, "accepted");
  assert.equal(refreshes, 1);
  assert.deepEqual(seen, ["old-token", "new-token"]);
});

test("does not loop when health returns the same rejected token", async () => {
  let requests = 0;
  await assert.rejects(
    withRefreshedRequestToken({
      token: "same-token",
      refreshToken: async () => "same-token",
      request: async () => {
        requests += 1;
        throw new LocalEngineRequestError("forbidden", 403);
      },
    }),
    /forbidden/,
  );
  assert.equal(requests, 1);
});

test("does not refresh non-authentication failures", async () => {
  await assert.rejects(
    withRefreshedRequestToken({
      token: "current-token",
      refreshToken: async () => {
        assert.fail("A server failure must not rotate credentials.");
      },
      request: async () => {
        throw new LocalEngineRequestError("queue full", 429);
      },
    }),
    /queue full/,
  );
});
